"""
AgentRunner: query entry point using the Claude Agent SDK.

Replaces the Plan-then-Execute architecture (Planner → Executor → Synthesis)
with a single agentic loop where Claude directly orchestrates all domain tools.

Uses ``ClaudeSDKClient`` (not ``query()``) because only the client supports
custom MCP tools.
"""

import asyncio
import logging
import os
import time
import traceback

from ct.agent.types import ExecutionResult, Plan, Step

logger = logging.getLogger("ct.runner")


# ------------------------------------------------------------------
# Testable message processing (extracted from _run_async)
# ------------------------------------------------------------------

async def process_messages(
    messages_iter,
    trace_renderer=None,
    headless=False,
    trace_events: list[dict] | None = None,
    thinking_status=None,
    runner=None,
    on_activity=None,
):
    """Process an async iterable of SDK messages into structured results.

    This is extracted from ``AgentRunner._run_async`` so it can be tested
    with mock message streams without a live SDK client.

    Args:
        messages_iter: Async iterable of SDK messages.
        trace_renderer: Optional TraceRenderer for console output.
        headless: If True, suppress console output.
        trace_events: Optional list to append trace events to. When provided,
            each TextBlock, ToolUseBlock, and ToolResultBlock produces a
            trace event dict for downstream notebook/export consumers.
        thinking_status: Optional ThinkingStatus to stop on first message.

    Returns:
        dict with keys: full_text, tool_calls, result_msg, streamed_len
    """
    # Lazy imports — these may not be available in unit tests without
    # the SDK installed, but callers pass mock objects anyway.
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            ToolResultBlock,
            StreamEvent,
        )
    except ImportError:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
        )
        ToolResultBlock = None
        StreamEvent = None

    full_text: list[str] = []
    tool_calls: list[dict] = []
    inflight: dict[str, dict] = {}  # tool_use_id → {name, input, start_time}
    result_msg = None
    streamed_len = 0  # characters already displayed via StreamEvent

    async for message in messages_iter:

        # --- StreamEvent (partial streaming) ---
        if StreamEvent is not None and isinstance(message, StreamEvent):
            event = getattr(message, "event", None) or {}
            if isinstance(event, dict):
                delta = event.get("delta", {})
                if isinstance(delta, dict) and delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        # Track streamed length but don't print raw text —
                        # the full TextBlock will be rendered as markdown
                        streamed_len += len(text)
            continue

        # --- AssistantMessage ---
        if isinstance(message, AssistantMessage):
            for block in (message.content or []):
                if isinstance(block, TextBlock):
                    # Stop the spinner when showing complete text block
                    if thinking_status is not None:
                        thinking_status.stop()
                        thinking_status = None
                        if runner is not None:
                            runner._active_spinner = None
                            if hasattr(runner, "session"):
                                runner.session._active_spinner = None
                        
                    text = block.text or ""
                    full_text.append(text)
                    # Trace capture
                    if trace_events is not None and text.strip():
                        trace_events.append({
                            "type": "text",
                            "content": text,
                            "timestamp": time.time(),
                        })
                    # Render as markdown (streamed deltas are tracked but not printed)
                    if not headless and trace_renderer:
                        streamed_len = 0  # reset for next turn
                        trace_renderer.render_reasoning(text)
                    # Activity callback — show snippet of reasoning
                    if on_activity and text.strip():
                        snippet = text.strip().replace("\n", " ")[:40]
                        on_activity(snippet)

                elif isinstance(block, ToolUseBlock):
                    # Restart spinner while waiting for tool result
                    if thinking_status is None and not headless and trace_renderer:
                        try:
                            from ct.ui.status import ThinkingStatus
                            thinking_status = ThinkingStatus(trace_renderer.console, phase="evaluating")
                            thinking_status.__enter__()
                            thinking_status.start_async_refresh()
                            if runner is not None:
                                runner._active_spinner = thinking_status
                            # Also store on session so MCP prompt can stop it
                            if runner is not None and hasattr(runner, "session"):
                                runner.session._active_spinner = thinking_status
                        except ImportError:
                            pass
                        
                    block_id = getattr(block, "id", "") or ""
                    now = time.time()
                    inflight[block_id] = {
                        "name": block.name,
                        "input": block.input,
                        "start_time": now,
                    }
                    tool_calls.append({
                        "name": block.name,
                        "input": block.input,
                    })
                    # Trace capture
                    if trace_events is not None:
                        trace_events.append({
                            "type": "tool_start",
                            "tool": block.name.replace("mcp__ct-tools__", ""),
                            "input": block.input,
                            "tool_use_id": block_id,
                            "timestamp": now,
                        })
                    if not headless and trace_renderer:
                        trace_renderer.render_tool_start(block.name, block.input)
                    # Activity callback — show tool name
                    if on_activity:
                        clean = block.name.replace("mcp__ct-tools__", "")
                        on_activity(f"\u25b8 {clean}")

                elif ToolResultBlock is not None and isinstance(block, ToolResultBlock):
                    tool_use_id = getattr(block, "tool_use_id", "") or ""
                    is_error = getattr(block, "is_error", False)

                    # Extract result text from content
                    content = getattr(block, "content", None)
                    result_text = ""
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                result_text += item.get("text", "")
                    elif isinstance(content, str):
                        result_text = content

                    # Match to inflight tracker
                    tracked = inflight.pop(tool_use_id, None)
                    duration = 0.0
                    tool_name = ""
                    tool_input = {}
                    if tracked:
                        duration = time.time() - tracked["start_time"]
                        tool_name = tracked["name"]
                        tool_input = tracked["input"]
                    else:
                        logger.warning(
                            "Orphan ToolResultBlock with tool_use_id=%s",
                            tool_use_id,
                        )

                    # Update the matching tool_calls entry with result
                    for tc in reversed(tool_calls):
                        if tc["name"] == tool_name and "result_text" not in tc:
                            tc["result_text"] = result_text
                            tc["duration_s"] = duration
                            break

                    # Trace capture
                    if trace_events is not None:
                        clean_tool = tool_name.replace("mcp__ct-tools__", "")
                        trace_events.append({
                            "type": "tool_result",
                            "tool": clean_tool,
                            "tool_use_id": tool_use_id,
                            "result_text": result_text,
                            "is_error": is_error,
                            "duration_s": duration,
                            "timestamp": time.time(),
                        })

                    if not headless and trace_renderer:
                        if is_error:
                            trace_renderer.render_tool_error(
                                tool_name or "unknown", result_text
                            )
                        else:
                            trace_renderer.render_tool_complete(
                                tool_name or "unknown",
                                tool_input,
                                result_text,
                                duration,
                            )

        # --- ResultMessage ---
        elif isinstance(message, ResultMessage):
            # Final message, make sure animation is stopped
            if thinking_status is not None:
                thinking_status.stop()
                thinking_status = None
                if runner is not None:
                    runner._active_spinner = None
                
            result_msg = message

    return {
        "full_text": full_text,
        "tool_calls": tool_calls,
        "result_msg": result_msg,
        "streamed_len": streamed_len,
    }


class AgentRunner:
    """Run queries via the Claude Agent SDK agentic loop.

    All 192 domain tools are exposed as MCP tools.  Claude handles planning,
    execution, error recovery, and synthesis in one conversation.
    """

    def __init__(
        self,
        session,
        trajectory=None,
        headless: bool = False,
        trace_store=None,
    ):
        self.session = session
        self.trajectory = trajectory
        self._headless = headless
        self.trace_store = trace_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        context: dict | None = None,
        progress_callback=None,
    ) -> ExecutionResult:
        """Execute a query synchronously (blocking wrapper around async)."""
        return asyncio.run(
            self._run_async(query, context, progress_callback)
        )

    async def _run_async(
        self,
        query: str,
        context: dict | None = None,
        progress_callback=None,
    ) -> ExecutionResult:
        """Execute a query using the Agent SDK agentic loop.

        Uses ``ClaudeSDKClient`` (bidirectional client) which supports custom
        MCP tools, unlike ``query()`` which does not.
        """
        from claude_agent_sdk import (
            ClaudeSDKClient,
            ClaudeAgentOptions,
        )

        # Start spinner immediately — user should see feedback the moment they hit Enter
        thinking_status = None
        if not self._headless:
            from ct.ui.status import ThinkingStatus
            thinking_status = ThinkingStatus(self.session.console, phase="planning")
            thinking_status.__enter__()
            thinking_status.start_async_refresh()
            self._active_spinner = thinking_status
        from ct.agent.mcp_server import create_ct_mcp_server
        from ct.agent.system_prompt import build_system_prompt
        from ct.ui.traces import TraceRenderer

        t0 = time.time()
        config = self.session.config
        ctx = context or {}

        # ----- Build MCP server -----
        exclude_cats = set()
        if not config.get("agent.enable_experimental_tools", False):
            from ct.tools import EXPERIMENTAL_CATEGORIES
            exclude_cats = set(EXPERIMENTAL_CATEGORIES)

        server, sandbox, tool_names, code_trace_buffer = create_ct_mcp_server(
            self.session,
            exclude_categories=exclude_cats,
        )

        # ----- Build system prompt -----
        data_context = None
        data_dir = ctx.get("data_dir")
        if data_dir:
            data_context = f"Data directory: {data_dir}\n"
            config.set("sandbox.extra_read_dirs", str(data_dir))

        history = None
        if self.trajectory and self.trajectory.turns:
            history = self.trajectory.context_for_planner()

        system_prompt = build_system_prompt(
            self.session,
            tool_names=tool_names,
            data_context=data_context,
            history=history,
        )

        # ----- Configure Agent SDK -----
        model = config.get("llm.model") or "claude-opus-4-6"
        max_turns = int(config.get("agent.max_sdk_turns", 30))

        allowed_tools = [f"mcp__ct-tools__{name}" for name in tool_names]

        _STRIP_VARS = {
            "CLAUDECODE",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDE_CODE_PARENT_SESSION_ID",
        }
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in _STRIP_VARS
        }
        api_key = config.llm_api_key("anthropic")
        if api_key:
            clean_env["ANTHROPIC_API_KEY"] = api_key
        # Suppress warnings in SDK subprocess (matplotlib, pydeseq2, numpy, etc.)
        clean_env["PYTHONWARNINGS"] = "ignore"

        # Enable Foundry mode for Agent SDK subprocess if Foundry env vars present
        if any(clean_env.get(v) for v in (
            "ANTHROPIC_FOUNDRY_API_KEY",
            "ANTHROPIC_FOUNDRY_RESOURCE",
            "ANTHROPIC_FOUNDRY_BASE_URL",
        )):
            clean_env["CLAUDE_CODE_USE_FOUNDRY"] = "1"
            # Pin model names for Foundry deployments
            clean_env.setdefault(
                "ANTHROPIC_DEFAULT_SONNET_MODEL", model
            )
            clean_env.setdefault(
                "ANTHROPIC_DEFAULT_OPUS_MODEL", model
            )
            clean_env.setdefault(
                "ANTHROPIC_DEFAULT_HAIKU_MODEL", model
            )

        # Plan mode: use SDK's built-in plan permission mode.
        # In plan mode, Claude outputs a plan then calls ExitPlanMode.
        # We intercept that to show the plan and ask for approval.
        plan_preview = bool(config.get("agent.plan_preview", False))
        permission_mode = "plan" if (plan_preview and not self._headless) else "bypassPermissions"

        # Enable streaming for real-time output
        options_kwargs = dict(
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            mcp_servers={"ct-tools": server},
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
            env=clean_env,
            hooks={},  # Disable inherited hooks (e.g. from Claude Code)
        )

        if plan_preview and not self._headless:
            options_kwargs["can_use_tool"] = self._plan_approval_hook()

        # Try to enable partial message streaming (graceful fallback)
        try:
            options = ClaudeAgentOptions(
                include_partial_messages=True,
                **options_kwargs,
            )
        except TypeError:
            # SDK version doesn't support include_partial_messages
            logger.info("SDK does not support include_partial_messages, using non-streaming")
            options = ClaudeAgentOptions(**options_kwargs)

        # ----- Build user prompt -----
        user_prompt = query
        context_parts = []
        if ctx.get("compound_smiles"):
            context_parts.append(f"Compound SMILES: {ctx['compound_smiles']}")
        if ctx.get("target"):
            context_parts.append(f"Target: {ctx['target']}")
        if ctx.get("indication"):
            context_parts.append(f"Indication: {ctx['indication']}")
        # Inject mention context if present
        if ctx.get("mention_context"):
            context_parts.append(ctx["mention_context"])
        if context_parts:
            user_prompt = query + "\n\nContext:\n" + "\n".join(context_parts)

        # ----- Create trace renderer -----
        trace_renderer = TraceRenderer(self.session.console)

        # ----- Prepare trace capture -----
        trace_events: list[dict] | None = None
        if self.trace_store is not None:
            trace_events = []

        # ----- Run the agentic loop via ClaudeSDKClient -----
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_prompt)
                result = await process_messages(
                    client.receive_response(),
                    trace_renderer=trace_renderer,
                    headless=self._headless,
                    trace_events=trace_events,
                    thinking_status=thinking_status,
                    runner=self,
                    on_activity=progress_callback,
                )
        except Exception as e:
            logger.error("Agent SDK query failed: %s\n%s", e, traceback.format_exc())
            duration = time.time() - t0
            return self._make_error_result(query, str(e), duration)
        finally:
            # Ensure animation is cleaned up even on error
            if thinking_status is not None:
                thinking_status.stop()

        duration = time.time() - t0

        full_text = result["full_text"]
        tool_calls = result["tool_calls"]
        result_msg = result["result_msg"]

        # ----- Build ExecutionResult -----
        summary = "\n".join(full_text).strip()
        if not summary:
            summary = "(Agent produced no text output)"

        answer = None
        if sandbox:
            result_var = sandbox.get_variable("result")
            if isinstance(result_var, dict):
                answer = result_var.get("answer")

        steps = []
        for i, tc in enumerate(tool_calls, 1):
            step = Step(
                id=i,
                tool=tc["name"].replace("mcp__ct-tools__", ""),
                description=f"Called {tc['name']}",
                tool_args=tc.get("input", {}),
            )
            step.status = "completed"
            steps.append(step)

        plan = Plan(query=query, steps=steps)

        cost_usd = 0.0
        if result_msg:
            cost_usd = getattr(result_msg, "total_cost_usd", 0.0) or 0.0

        exec_result = ExecutionResult(
            plan=plan,
            summary=summary,
            raw_results={"tool_calls": tool_calls, "answer": answer},
            duration_s=duration,
            iterations=1,
            metadata={
                "sdk_cost_usd": cost_usd,
                "sdk_turns": getattr(result_msg, "num_turns", 0) if result_msg else 0,
                "sdk_duration_ms": getattr(result_msg, "duration_ms", 0) if result_msg else 0,
                "tool_call_count": len(tool_calls),
            },
        )

        # ----- Inject tool_result events from code_trace_buffer -----
        # The SDK stream typically does NOT include ToolResultBlock messages,
        # so process_messages() only produces tool_start events for code tools.
        # MCP handlers write structured results (code, stdout, plots) to
        # code_trace_buffer. We match buffer entries to tool_start events
        # by tool name in sequential order, and insert tool_result events
        # immediately after each tool_start.
        if trace_events is not None and trace_events:
            buffer_iter = iter(code_trace_buffer)
            # Also create tool_result events for non-code tools from tool_calls
            non_code_results = {}
            for tc in tool_calls:
                name = tc["name"].replace("mcp__ct-tools__", "")
                if name not in ("run_python", "run_r") and "result_text" in tc:
                    key = name + ":" + str(tc.get("input", {}))
                    non_code_results[key] = tc

            enriched: list[dict] = []
            non_code_iter_idx = {}  # track which non-code tool_calls we've used
            for event in trace_events:
                enriched.append(event)
                if event.get("type") != "tool_start":
                    continue

                tool = event.get("tool", "")
                tool_use_id = event.get("tool_use_id", "")

                if tool in ("run_python", "run_r"):
                    meta = next(buffer_iter, None)
                    if meta:
                        enriched.append({
                            "type": "tool_result",
                            "tool": tool,
                            "tool_use_id": tool_use_id,
                            "result_text": meta.get("stdout", ""),
                            "is_error": bool(meta.get("error")),
                            "duration_s": 0.0,
                            "code": meta.get("code", ""),
                            "stdout": meta.get("stdout", ""),
                            "plots": meta.get("plots", []),
                            "exports": meta.get("exports", []),
                            "error": meta.get("error"),
                            "timestamp": time.time(),
                        })
                else:
                    # For non-code tools, find matching result from tool_calls
                    for tc in tool_calls:
                        tc_name = tc["name"].replace("mcp__ct-tools__", "")
                        if tc_name == tool and "result_text" in tc and not tc.get("_used"):
                            tc["_used"] = True
                            enriched.append({
                                "type": "tool_result",
                                "tool": tool,
                                "tool_use_id": tool_use_id,
                                "result_text": tc["result_text"],
                                "is_error": False,
                                "duration_s": tc.get("duration_s", 0.0),
                                "timestamp": time.time(),
                            })
                            break

            trace_events = enriched

        # ----- Flush trace events -----
        if self.trace_store is not None and trace_events:
            try:
                self.trace_store.add_events(
                    trace_events,
                    query=query,
                    model=model,
                    duration_s=duration,
                    cost_usd=cost_usd,
                )
                self.trace_store.flush()
            except Exception as e:
                logger.warning("Failed to flush trace: %s", e)

        if not self._headless and result_msg:
            self._print_usage(result_msg, duration)

        return exec_result

    # ------------------------------------------------------------------
    # Plan mode
    # ------------------------------------------------------------------

    def _plan_approval_hook(self):
        """Return a can_use_tool callback for SDK plan mode.

        Intercepts the ExitPlanMode call to show Claude's plan and ask
        for user approval. All other tool calls are auto-allowed.
        """
        console = self.session.console
        # Shared ref so process_messages can keep it in sync
        self._active_spinner = None

        async def _hook(tool_name, input_data, context):
            if tool_name == "ExitPlanMode":
                # Stop the spinner so it doesn't interfere with input()
                if self._active_spinner is not None:
                    self._active_spinner.stop()
                    self._active_spinner = None

                # Claude is requesting to exit plan mode and start executing
                console.print("\n  [bold cyan]Proposed Plan[/bold cyan]")
                # The plan text may be in the input data or in Claude's
                # preceding text output (which the user already saw streamed).
                if isinstance(input_data, dict):
                    for key in ("plan", "description", "summary"):
                        if key in input_data and input_data[key]:
                            console.print(f"  {input_data[key]}")
                            break
                console.print()

                try:
                    answer = input("  Execute this plan? [Y/n] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"

                if answer in ("", "y", "yes"):
                    return {"allow": True, "updated_input": input_data}
                else:
                    # Ask what to change so Claude can revise the plan
                    try:
                        feedback = input("  What would you change? ").strip()
                    except (EOFError, KeyboardInterrupt):
                        feedback = ""

                    msg = f"User rejected the plan. Feedback: {feedback}" if feedback else "User rejected the plan."
                    return {"allow": False, "message": msg}

            # All other tools: allow
            return {"allow": True, "updated_input": input_data}

        return _hook

    # ------------------------------------------------------------------
    # Console output helpers
    # ------------------------------------------------------------------

    def _print_usage(self, result_msg, duration: float):
        """Print cost and usage summary."""
        cost = getattr(result_msg, "total_cost_usd", 0)
        turns = getattr(result_msg, "num_turns", 0)
        parts = []
        if cost:
            parts.append(f"${cost:.2f}")
        if turns:
            parts.append(f"{turns} turns")
        if duration >= 60:
            mins = int(duration // 60)
            secs = int(duration % 60)
            parts.append(f"{mins}m {secs}s")
        else:
            parts.append(f"{duration:.1f}s")
        self.session.console.print(f"\n  [dim]{' | '.join(parts)}[/dim]")

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _make_error_result(query: str, error: str, duration: float) -> ExecutionResult:
        """Build an ExecutionResult representing a failed query."""
        plan = Plan(query=query, steps=[])
        return ExecutionResult(
            plan=plan,
            summary=f"Agent SDK error: {error}",
            raw_results={"error": error},
            duration_s=duration,
            iterations=1,
        )
