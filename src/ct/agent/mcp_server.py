"""
MCP tool server for the Claude Agent SDK.

Wraps the existing ct ToolRegistry so every registered tool is exposed as an
MCP tool that the Agent SDK can invoke. Also provides a persistent ``run_python``
sandbox tool for multi-turn code execution.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

logger = logging.getLogger("ct.mcp_server")


# ---------------------------------------------------------------------------
# Tool result formatting
# ---------------------------------------------------------------------------

def _format_tool_result(result: Any, max_chars: int = 8000) -> str:
    """Format a ct tool result dict into text for the Agent SDK."""
    if not isinstance(result, dict):
        text = str(result)
        return text[:max_chars] if len(text) > max_chars else text

    parts = []

    # Summary first (most important)
    summary = result.get("summary", "")
    if summary:
        parts.append(summary)

    # Include key data fields
    skip = {"summary"}
    compact = {"top_hits", "top_terms"}
    for key, val in result.items():
        if key in skip:
            continue
        if key in compact and isinstance(val, (dict, list)):
            count = len(val)
            parts.append(f"{key}: {type(val).__name__} with {count} entries")
            continue
        val_str = str(val)
        if len(val_str) > 1500:
            val_str = val_str[:1500] + f"... [{len(val_str)} chars total]"
        parts.append(f"{key}: {val_str}")

    text = "\n".join(parts)
    return text[:max_chars] if len(text) > max_chars else text


# ---------------------------------------------------------------------------
# Convert ct Tool.parameters to JSON Schema
# ---------------------------------------------------------------------------

_PY_TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "dict": "object",
}


def _params_to_json_schema(parameters: dict) -> dict:
    """Convert a ct tool parameters dict to a JSON Schema object.

    ct tools describe parameters as ``{param_name: description_string}``.
    We map these to string-typed JSON Schema properties since the LLM
    produces string values that tools coerce internally.
    """
    if not parameters:
        return {"type": "object", "properties": {}}

    properties = {}
    for name, desc in parameters.items():
        # Extract type hint from description if present (e.g., "gene name (str)")
        prop = {"type": "string", "description": str(desc)}
        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
    }


# ---------------------------------------------------------------------------
# Create MCP tools from registry
# ---------------------------------------------------------------------------

def _make_tool_handler(tool_obj, session):
    """Create an async handler for a registry tool.

    The handler runs the synchronous tool function in a thread to avoid
    blocking the event loop.
    """

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        # Inject session and prior results (empty for SDK mode)
        call_args = dict(args)
        call_args["_session"] = session
        call_args["_prior_results"] = {}

        # Coerce string values to numeric types when they look like numbers.
        # MCP sends all parameters as strings, but tools often expect int/float.
        for key, val in list(call_args.items()):
            if key.startswith("_"):
                continue
            if isinstance(val, str):
                # Try int first, then float
                try:
                    call_args[key] = int(val)
                    continue
                except ValueError:
                    pass
                try:
                    call_args[key] = float(val)
                    continue
                except ValueError:
                    pass
                # Boolean coercion
                if val.lower() in ("true", "false"):
                    call_args[key] = val.lower() == "true"

        try:
            # Route GPU tools through compute router
            if getattr(tool_obj, "requires_gpu", False):
                from ct.cloud.router import ComputeRouter
                router = ComputeRouter(config=getattr(session, "config", None))
                result = await asyncio.to_thread(router.route, tool_obj, **call_args)

                # Handle "needs_user_prompt" — the router couldn't run locally
                # and needs the user to decide about cloud fallback.
                # We prompt here on the main thread where input() works.
                if isinstance(result, dict) and result.get("needs_user_prompt"):
                    result = await _prompt_cloud_fallback(
                        router, tool_obj, result, call_args, session,
                    )
            else:
                result = await asyncio.to_thread(tool_obj.run, **call_args)
            text = _format_tool_result(result)
        except Exception as e:
            logger.warning("Tool %s raised: %s", tool_obj.name, e)
            text = f"Error: {e}"
            return {
                "content": [{"type": "text", "text": text}],
                "is_error": True,
            }

        return {"content": [{"type": "text", "text": text}]}

    return handler


async def _prompt_cloud_fallback(router, tool_obj, prompt_result, call_args, session):
    """Prompt the user on the main thread about switching to CellType Cloud.

    Called when compute.mode=local but the tool can't run locally.
    Runs on the event loop (main thread) so input() works correctly.

    Options:
      y   — run this tool on CellType Cloud (one-time)
      n   — skip this tool
      n!  — skip and don't ask again for this session
    """
    from rich.console import Console
    console = getattr(session, "console", None) or Console()

    # Stop the spinner so it doesn't interfere with the prompt
    spinner = getattr(session, "_active_spinner", None)
    if spinner is not None:
        spinner.stop()
        session._active_spinner = None

    msg = prompt_result.get("prompt_message", "GPU tool cannot run locally.")

    console.print(f"\n  [yellow]{msg}[/yellow]")

    try:
        answer = input(
            "  Use CellType Cloud for this tool call?\n"
            "  [y] yes  [y!] yes for all (switch to cloud mode)  [n] no (skip this tool call): "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer in ("y", "yes"):
        # One-time cloud dispatch for this tool call
        result = await asyncio.to_thread(router.route_cloud_for_tool, tool_obj, **call_args)
        return result
    elif answer in ("y!", "yes!"):
        # Switch to cloud mode permanently — no more prompts
        try:
            from ct.agent.config import Config
            cfg = Config.load()
            cfg.set("compute.mode", "cloud")
            cfg.save()
            console.print("  [green]compute.mode switched to cloud[/green]")
        except Exception:
            pass
        result = await asyncio.to_thread(router.route_cloud_for_tool, tool_obj, **call_args)
        return result
    else:
        # n / anything else — skip this tool call
        return {
            "summary": (
                f"[Skipped] {tool_obj.name} — {msg} "
                "User chose to skip this tool call."
            ),
            "skipped": True,
            "reason": "user_skipped",
        }


# ---------------------------------------------------------------------------
# run_python sandbox tool
# ---------------------------------------------------------------------------

def _make_run_python_handler(session, code_trace_buffer: list | None = None):
    """Create the run_python MCP tool handler with a persistent Sandbox.

    The sandbox is created lazily on first invocation and persists across
    tool calls within one query (variables carry over). A new MCP server
    (and thus a new sandbox) is created for each query, so state resets
    between queries automatically.

    Args:
        session: Active ct Session.
        code_trace_buffer: Optional shared list. When provided, the handler
            appends structured execution metadata after each call. This
            bypasses the SDK message stream (which may truncate tool results)
            so the trace collector gets full code, stdout, and plot data.
    """
    from ct.agent.sandbox import Sandbox

    config = session.config
    timeout = int(config.get("sandbox.timeout", 300))
    output_dir = config.get("sandbox.output_dir")
    max_retries = int(config.get("sandbox.max_retries", 2))

    extra_read_dirs = []
    extra_read_str = config.get("sandbox.extra_read_dirs")
    if extra_read_str:
        for d in str(extra_read_str).split(","):
            d = d.strip()
            if d and Path(d).exists():
                extra_read_dirs.append(Path(d))

    sandbox = Sandbox(
        timeout=timeout,
        output_dir=output_dir,
        max_retries=max_retries,
        extra_read_dirs=extra_read_dirs or None,
    )
    sandbox.load_datasets()

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        code = args.get("code", "")
        if not code.strip():
            return {
                "content": [{"type": "text", "text": "Error: no code provided"}],
                "is_error": True,
            }

        exec_result = await asyncio.to_thread(sandbox.execute, code)

        # Build output text
        parts = []
        if exec_result.get("stdout"):
            parts.append(exec_result["stdout"])
        if exec_result.get("error"):
            parts.append(f"Error:\n{exec_result['error']}")
        if exec_result.get("plots"):
            parts.append(f"Plots saved: {exec_result['plots']}")
        if exec_result.get("exports"):
            parts.append(f"Exports saved: {exec_result['exports']}")

        # Check if the code set a `result` variable
        result_var = sandbox.get_variable("result")
        if result_var and isinstance(result_var, dict):
            summary = result_var.get("summary", "")
            answer = result_var.get("answer", "")
            if summary:
                parts.append(f"\nResult summary: {summary}")
            if answer:
                parts.append(f"Result answer: {answer}")

        text = "\n".join(parts) if parts else "(no output)"
        # Cap output to keep context manageable
        text = text[:6000]

        # Buffer structured execution metadata for trace capture.
        # This bypasses the SDK stream which may truncate tool results.
        if code_trace_buffer is not None:
            code_trace_buffer.append({
                "tool": "run_python",
                "code": code,
                "stdout": exec_result.get("stdout", ""),
                "plots": exec_result.get("plots", []),
                "exports": exec_result.get("exports", []),
                "error": exec_result.get("error"),
            })

        is_error = bool(exec_result.get("error"))
        return {
            "content": [{"type": "text", "text": text}],
            "is_error": is_error,
        }

    return handler, sandbox


# ---------------------------------------------------------------------------
# run_r tool — first-class R execution via rpy2
# ---------------------------------------------------------------------------

def _make_run_r_handler(code_trace_buffer: list | None = None):
    """Create the run_r MCP tool handler for R code execution.

    Uses rpy2 to execute R code. The global R session persists across calls
    (packages stay loaded, variables carry over within a query).
    """

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        code = args.get("code", "")
        if not code.strip():
            return {
                "content": [{"type": "text", "text": "Error: no R code provided"}],
                "is_error": True,
            }

        def _exec_r(code: str) -> str:
            try:
                import rpy2.robjects as ro
                from rpy2.robjects import numpy2ri, pandas2ri

                # Use capture.output to grab all printed/cat output
                # Wrap user code in braces so multi-line code works
                escaped = code.replace("\\", "\\\\").replace("'", "\\'")
                wrapper = f"paste(capture.output({{ {code} }}), collapse='\\n')"

                try:
                    captured = ro.r(wrapper)
                    output_text = str(captured[0]) if captured else ""
                except Exception:
                    # If capture.output fails (syntax error etc), run directly
                    # to get the actual R error message
                    try:
                        result = ro.r(code)
                        output_text = str(result)[:3000]
                    except Exception as e2:
                        return f"R Error: {e2}"

                # Also get the return value of the last expression
                # by running the code directly (capture.output eats return values)
                result_text = ""
                try:
                    result = ro.r(code)
                    if result is not None and result != ro.NULL:
                        numpy2ri.activate()
                        pandas2ri.activate()
                        try:
                            if hasattr(result, '__len__') and len(result) == 1:
                                result_text = f"\nReturn value: {float(result[0])}"
                            elif hasattr(result, '__len__') and len(result) <= 50:
                                vals = [str(x) for x in result]
                                result_text = f"\nReturn value: [{', '.join(vals)}]"
                            else:
                                result_text = f"\nReturn value: {str(result)[:2000]}"
                        except Exception:
                            result_text = f"\nReturn value: {str(result)[:2000]}"
                        finally:
                            numpy2ri.deactivate()
                            pandas2ri.deactivate()
                except Exception:
                    pass  # Already captured output above

                return (output_text + result_text).strip() or "(no output)"

            except Exception as e:
                return f"R Error: {e}"

        text = await asyncio.to_thread(_exec_r, code)
        text = text[:6000]
        is_error = text.startswith("R Error:")

        # Buffer structured execution metadata for trace capture.
        if code_trace_buffer is not None:
            code_trace_buffer.append({
                "tool": "run_r",
                "code": code,
                "stdout": text,
                "error": text if is_error else None,
            })

        return {
            "content": [{"type": "text", "text": text}],
            "is_error": is_error,
        }

    return handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_ct_mcp_server(
    session,
    *,
    exclude_categories: set[str] | None = None,
    exclude_tools: set[str] | None = None,
    include_run_python: bool = True,
):
    """Create an in-process MCP server exposing all ct tools.

    Args:
        session: Active ct Session (provides config, LLM client).
        exclude_categories: Tool categories to omit.
        exclude_tools: Specific tool names to omit.
        include_run_python: Whether to include the run_python sandbox tool.

    Returns:
        A tuple of ``(mcp_server, sandbox_or_none, tool_names, code_trace_buffer)``
        where sandbox is the Sandbox instance (if run_python is enabled) for
        post-query inspection, and code_trace_buffer is a shared list that
        MCP handlers append structured execution metadata to.
    """
    from ct.tools import registry, ensure_loaded, EXPERIMENTAL_CATEGORIES

    ensure_loaded()

    exclude_categories = exclude_categories or set()
    exclude_tools = exclude_tools or set()

    # Shared buffer: code tool handlers append structured metadata here.
    # The runner reads from this to enrich trace events — bypasses the SDK
    # stream which may truncate/omit tool result content.
    code_trace_buffer: list[dict] = []

    sdk_tools: list[SdkMcpTool] = []
    tool_names: list[str] = []

    for tool_obj in registry.list_tools():
        if tool_obj.category in exclude_categories:
            continue
        if tool_obj.name in exclude_tools:
            continue
        # Skip experimental categories by default
        if tool_obj.category in EXPERIMENTAL_CATEGORIES:
            continue

        handler = _make_tool_handler(tool_obj, session)
        schema = _params_to_json_schema(tool_obj.parameters)

        sdk_tool = SdkMcpTool(
            name=tool_obj.name,
            description=tool_obj.description,
            input_schema=schema,
            handler=handler,
        )
        sdk_tools.append(sdk_tool)
        tool_names.append(tool_obj.name)

    # Add run_python tool
    sandbox = None
    if include_run_python:
        rp_handler, sandbox = _make_run_python_handler(session, code_trace_buffer)
        rp_tool = SdkMcpTool(
            name="run_python",
            description=(
                "Execute Python code in a sandboxed environment. Variables persist "
                "between calls. Pre-imported: pd, np, plt, sns, scipy_stats, sklearn, "
                "json, re, math, collections, itertools, os, glob, gzip, csv, zipfile, "
                "io, tempfile, struct, datetime, Path, safe_subprocess_run, "
                "compute_pi_percentage, run_r (R via rpy2). "
                "Save plots to OUTPUT_DIR. When done, assign "
                "result = {'summary': '...', 'answer': '...'}"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
            handler=rp_handler,
        )
        sdk_tools.append(rp_tool)
        tool_names.append("run_python")

    # Add run_r tool (R code execution via rpy2)
    if include_run_python:  # R tool follows same gating as Python
        try:
            import rpy2.robjects  # noqa: F401 — check availability
            rr_handler = _make_run_r_handler(code_trace_buffer)
            rr_tool = SdkMcpTool(
                name="run_r",
                description=(
                    "Execute R code via rpy2. Use for: natural splines (ns()), "
                    "wilcox.test(), p.adjust(), fisher.test(), lm(), predict(), "
                    "survival analysis, KEGG pathway analysis (KEGGREST), and any "
                    "analysis where R is the reference implementation. "
                    "Available packages: stats, splines, survival, MASS, KEGGREST. "
                    "Print results with cat() or print(). "
                    "Use this instead of run_python when the question asks for R, or when "
                    "R's implementation is the reference (splines, multiple testing correction, "
                    "nonparametric tests, organism-specific KEGG ORA)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "R code to execute",
                        }
                    },
                    "required": ["code"],
                },
                handler=rr_handler,
            )
            sdk_tools.append(rr_tool)
            tool_names.append("run_r")
        except ImportError:
            logger.info("rpy2 not available — run_r tool disabled")

    server = create_sdk_mcp_server(
        name="ct-tools",
        version="1.0.0",
        tools=sdk_tools,
    )

    logger.info(
        "Created MCP server with %d tools (%d domain + %s)",
        len(sdk_tools),
        len(sdk_tools) - (1 if include_run_python else 0) - (1 if "run_r" in tool_names else 0),
        ", ".join(t for t in ["run_python", "run_r"] if t in tool_names) or "no sandbox",
    )

    return server, sandbox, tool_names, code_trace_buffer
