"""Tests for contextual ghost-text suggestions and terminal features."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent
from ct.ui.terminal import (
    _extract_llm_suggestions,
    extract_mentions,
    build_mention_context,
    MentionCompleter,
    MergedCompleter,
    SlashCompleter,
    DATASET_CANDIDATES,
    KNOWN_DATASETS,
    DEFAULT_SUGGESTIONS,
    SLASH_COMMANDS,
    InteractiveTerminal,
)



class TestSlashCommands:
    """Verify all expected slash commands are registered."""

    def test_core_commands_registered(self):
        expected = ["/help", "/tools", "/model", "/settings", "/usage", "/copy",
                    "/export", "/compact", "/sessions", "/resume",
                    "/clear", "/exit", "/config", "/keys", "/doctor"]
        for cmd in expected:
            assert cmd in SLASH_COMMANDS, f"{cmd} not in SLASH_COMMANDS"

    def test_slash_commands_have_descriptions(self):
        for cmd, desc in SLASH_COMMANDS.items():
            assert isinstance(desc, str) and len(desc) > 5, f"{cmd} has no description"



class TestTerminalMethods:
    """Unit tests for InteractiveTerminal methods (no actual REPL)."""

    @pytest.fixture
    def terminal(self):
        """Create a terminal with mocked session."""
        with patch("ct.ui.terminal.InteractiveTerminal.__init__", return_value=None):
            t = InteractiveTerminal.__new__(InteractiveTerminal)
            t.session = MagicMock()
            t.session.verbose = False
            t.session.current_model = "claude-sonnet-4-5-20250929"
            t.session.config = MagicMock()
            t.session.config.get.side_effect = lambda key, default=None: {
                "agent.plan_preview": False,
            }.get(key, default)
            t.console = MagicMock()
            t._last_response = None
            t._verbose_hint = None
            t._show_exit_hint = False
            t._suggestions = list(DEFAULT_SUGGESTIONS)
            t._suggestion_idx = 0
            return t

    def test_model_display_name(self, terminal):
        assert terminal._model_display_name("claude-sonnet-4-5-20250929") == "Sonnet 4.5"
        assert terminal._model_display_name("claude-opus-4-6") == "Opus 4.6"
        assert terminal._model_display_name("unknown-model") == "unknown-model"

    def test_copy_no_response(self, terminal):
        terminal._copy_last_response()
        terminal.console.print.assert_called_once()
        assert "No response" in str(terminal.console.print.call_args)

    def test_copy_with_response(self, terminal):
        terminal._last_response = "GSPT1 is a promising target."
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            terminal._copy_last_response()
            mock_run.assert_called_once()
            assert mock_run.call_args[1]["input"] == b"GSPT1 is a promising target."

    def test_copy_clipboard_unavailable(self, terminal):
        terminal._last_response = "Some text"
        with patch("subprocess.run", side_effect=FileNotFoundError):
            terminal._copy_last_response()
            assert "not available" in str(terminal.console.print.call_args).lower()

    def test_bottom_toolbar_shows_gpu_tools_celltype_cloud(self, terminal):
        toolbar = terminal._bottom_toolbar()
        rendered = str(toolbar)
        assert "GPU tools:" in rendered
        assert "CellType Cloud" in rendered

    def test_export_no_session(self, terminal):
        terminal._export_session()
        terminal.console.print.assert_called_once()
        assert "No session" in str(terminal.console.print.call_args)

    def test_export_with_trajectory(self, terminal, tmp_path):
        """Export should write a markdown file."""
        from ct.agent.trajectory import Turn, Trajectory
        terminal.agent = MagicMock()
        terminal.agent.trajectory = Trajectory()
        terminal.agent.trajectory.turns = [
            Turn(query="test query", answer="test answer",
                 entities=["TP53"], tools_used=["target.coessentiality"],
                 timestamp=0),
        ]
        with patch("pathlib.Path.home", return_value=tmp_path):
            terminal._export_session()
            exports = list((tmp_path / ".ct" / "exports").glob("*.md"))
            assert len(exports) == 1
            content = exports[0].read_text()
            assert "test query" in content
            assert "test answer" in content
            assert "TP53" in content

    def test_run_shell_basic(self, terminal):
        terminal._run_shell("echo hello")
        # Should have printed the output
        terminal.console.print.assert_called()
        output = str(terminal.console.print.call_args_list[0])
        assert "hello" in output

    def test_run_shell_empty(self, terminal):
        terminal._run_shell("")
        terminal.console.print.assert_called_once()
        assert "Usage" in str(terminal.console.print.call_args)

    def test_run_shell_timeout(self, terminal):
        with patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("cmd", 30)):
            terminal._run_shell("sleep 100")
            assert "timed out" in str(terminal.console.print.call_args).lower()

    def test_run_shell_blocked(self, terminal):
        terminal._run_shell("rm -rf /tmp/x")
        terminal.console.print.assert_called_once()
        assert "blocked" in str(terminal.console.print.call_args).lower()

    def test_advance_suggestion(self, terminal):
        assert terminal._suggestion_idx == 0
        terminal._advance_suggestion()
        assert terminal._suggestion_idx == 1

    def test_update_suggestions_with_llm_result(self, terminal):
        """_update_suggestions extracts LLM suggestions from result.summary."""
        result = MagicMock()
        result.summary = (
            "## Suggested Next Steps\n"
            "- **\"Check co-essential partners of BRCA1\"**\n"
            "- **\"Run pathway enrichment on BRCA1 signature\"**\n"
        )
        terminal._update_suggestions("analyze BRCA1", result=result)
        assert len(terminal._suggestions) == 2
        assert any("BRCA1" in s for s in terminal._suggestions)

    def test_compact_context_calls_llm_chat_with_messages(self, terminal):
        from ct.agent.trajectory import Trajectory, Turn

        terminal.agent = MagicMock()
        trajectory = Trajectory()
        trajectory.turns = [
            Turn(query="q1", answer="a1"),
            Turn(query="q2", answer="a2"),
            Turn(query="q3", answer="a3"),
        ]
        terminal.agent.trajectory = trajectory

        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(content="session summary text")
        terminal.session.get_llm.return_value = mock_llm

        terminal._compact_context("focus on actionable outcomes")

        kwargs = mock_llm.chat.call_args.kwargs
        assert kwargs["system"].startswith("You are a research session summarizer")
        assert kwargs["messages"][0]["role"] == "user"
        assert "focus on actionable outcomes" in kwargs["messages"][0]["content"]

        assert len(terminal.agent.trajectory.turns) == 2
        assert terminal.agent.trajectory.turns[0].query == "[session summary]"
        assert terminal.agent.trajectory.turns[0].answer == "session summary text"

    def test_run_orchestrated_updates_suggestions_from_result(self, terminal):
        terminal.agent = MagicMock()
        terminal._update_suggestions = MagicMock()

        result = MagicMock()
        result.summary = "summary"
        result.merged_plan = MagicMock()

        with patch("ct.agent.orchestrator.ResearchOrchestrator") as mock_orchestrator:
            mock_orchestrator.return_value.run.return_value = result
            terminal._run_orchestrated("query", {}, 2)

        terminal._update_suggestions.assert_called_once_with("query", result.merged_plan, result)

    def test_case_study_updates_suggestions_from_result(self, terminal):
        terminal._update_suggestions = MagicMock()

        result = MagicMock()
        result.summary = "summary"
        result.merged_plan = MagicMock()

        case = MagicMock()
        case.name = "Demo Case"
        case.description = "desc"
        case.compound = "lenalidomide"

        with patch("ct.agent.case_studies.CASE_STUDIES", {"demo": case}), patch(
            "ct.agent.case_studies.run_case_study",
            return_value=result,
        ):
            terminal._handle_case_study_command("/case-study demo", {})

        terminal._update_suggestions.assert_called_once_with(case.compound, result.merged_plan, result)


class TestExtractMentions:
    """Tests for extract_mentions() — parsing @ tokens from query text."""

    def test_tool_mention(self):
        query, tools, datasets, workflows = extract_mentions("analyze CRBN with @target.coessentiality")
        assert "target.coessentiality" in tools
        assert datasets == []
        assert workflows == []
        assert "@" not in query

    def test_dataset_mention(self):
        query, tools, datasets, workflows = extract_mentions("check sensitivity in @depmap @prism")
        assert set(datasets) == {"depmap", "prism"}
        assert "@" not in query

    def test_mixed_mentions(self):
        query, tools, datasets, workflows = extract_mentions(
            "@depmap find @target.coessentiality for CRBN"
        )
        assert "target.coessentiality" in tools
        assert "depmap" in datasets
        assert "CRBN" in query

    def test_no_mentions(self):
        query, tools, datasets, workflows = extract_mentions("analyze CRBN degradation")
        assert query == "analyze CRBN degradation"
        assert tools == []
        assert datasets == []
        assert workflows == []

    def test_workflow_mention(self):
        query, tools, datasets, workflows = extract_mentions("run @target_validation on KRAS")
        assert "target_validation" in workflows
        assert tools == []
        assert datasets == []
        assert "KRAS" in query

    def test_mixed_with_workflow(self):
        query, tools, datasets, workflows = extract_mentions(
            "@depmap @target_validation check CRBN"
        )
        assert "depmap" in datasets
        assert "target_validation" in workflows
        assert "CRBN" in query


class TestBuildMentionContext:
    """Tests for build_mention_context() — formatting planner instructions."""

    def test_tools_only(self):
        ctx = build_mention_context(["target.coessentiality"], [])
        assert "target.coessentiality" in ctx
        assert "tool" in ctx.lower()

    def test_datasets_only(self):
        ctx = build_mention_context([], ["depmap", "prism"])
        assert "depmap" in ctx
        assert "prism" in ctx

    def test_empty(self):
        ctx = build_mention_context([], [])
        assert ctx == ""

    def test_workflow(self):
        ctx = build_mention_context([], [], ["target_validation"])
        assert "target_validation" in ctx
        assert "workflow" in ctx.lower()

    def test_mixed_with_workflow(self):
        ctx = build_mention_context(["target.coessentiality"], ["depmap"], ["compound_safety"])
        assert "target.coessentiality" in ctx
        assert "depmap" in ctx
        assert "compound_safety" in ctx


class TestMentionCompleterTabs:
    """Tests for MentionCompleter tabbed filtering."""

    @pytest.fixture
    def candidates(self):
        return [
            ("target.coessentiality", "target", "Co-essential gene networks", "tool"),
            ("expression.pathway_enrichment", "expression", "Pathway enrichment analysis", "tool"),
            ("depmap", "dataset", "DepMap CRISPR/model data", "dataset"),
            ("prism", "dataset", "PRISM drug sensitivity", "dataset"),
            ("CRISPRGeneEffect.csv", "file", "depmap/CRISPRGeneEffect.csv", "file"),
            ("target_validation", "workflow", "Validate a potential drug target (5 steps)", "workflow"),
        ]

    @pytest.fixture
    def completer(self, candidates):
        return MentionCompleter(candidates)

    def _get_completions(self, completer, text):
        doc = Document(text, len(text))
        return list(completer.get_completions(doc, CompleteEvent()))

    def test_all_tab_shows_everything(self, completer):
        completer._active_tab = 0
        completions = self._get_completions(completer, "@")
        assert len(completions) == 6

    def test_tools_tab_filters(self, completer):
        completer._active_tab = 1
        completions = self._get_completions(completer, "@")
        assert len(completions) == 2
        assert all("tool" in c.style for c in completions)

    def test_db_tab_filters(self, completer):
        completer._active_tab = 2
        completions = self._get_completions(completer, "@")
        assert len(completions) == 2
        names = {c.text for c in completions}
        assert names == {"@depmap", "@prism"}

    def test_files_tab_filters(self, completer):
        completer._active_tab = 3
        completions = self._get_completions(completer, "@")
        assert len(completions) == 1
        assert completions[0].text == "@CRISPRGeneEffect.csv"

    def test_flows_tab_filters(self, completer):
        completer._active_tab = 4  # Flows
        completions = self._get_completions(completer, "@")
        assert len(completions) == 1
        assert completions[0].text == "@target_validation"
        assert "workflow" in completions[0].style

    def test_substring_match(self, completer):
        completions = self._get_completions(completer, "@coess")
        assert len(completions) == 1
        assert completions[0].text == "@target.coessentiality"

    def test_no_at_returns_nothing(self, completer):
        completions = self._get_completions(completer, "hello world")
        assert len(completions) == 0


class TestMergedCompleterDispatch:
    """Tests for MergedCompleter dispatch logic."""

    @pytest.fixture
    def merged(self):
        candidates = [("depmap", "dataset", "DepMap data", "dataset")]
        return MergedCompleter(SlashCompleter(), MentionCompleter(candidates))

    def _get_completions(self, completer, text):
        doc = Document(text, len(text))
        return list(completer.get_completions(doc, CompleteEvent()))

    def test_slash_routes_to_slash(self, merged):
        completions = self._get_completions(merged, "/he")
        assert any(c.text == "/help" for c in completions)

    def test_at_routes_to_mention(self, merged):
        completions = self._get_completions(merged, "query @dep")
        assert any("depmap" in c.display_text for c in completions)

    def test_no_trigger_yields_nothing(self, merged):
        completions = self._get_completions(merged, "analyze compound")
        assert len(completions) == 0
