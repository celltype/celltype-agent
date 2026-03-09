"""Tests for CLI argument parsing and subcommand dispatch."""

import subprocess
from unittest.mock import patch

from typer.testing import CliRunner

from ct.agent.config import Config
from ct.cli import app


runner = CliRunner()


def test_keys_subcommand_not_treated_as_query():
    with patch("ct.cli.run_query") as mock_run_query, patch(
        "ct.agent.config.Config.load", return_value=Config(data={})
    ):
        result = runner.invoke(app, ["keys"])

    assert result.exit_code == 0
    assert "API Keys" in result.stdout
    mock_run_query.assert_not_called()


def test_doctor_subcommand_not_treated_as_query():
    with patch("ct.cli.run_query") as mock_run_query, patch(
        "ct.agent.config.Config.load", return_value=Config(data={"llm.api_key": "x"})
    ):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "ct Doctor" in result.stdout
    mock_run_query.assert_not_called()


def test_query_mode_uses_remaining_args_as_query():
    with patch("ct.cli.run_query") as mock_run_query, patch(
        "ct.cli.run_interactive"
    ) as mock_run_interactive:
        result = runner.invoke(app, ["run", "profile", "TP53", "in", "AML"])

    assert result.exit_code == 0
    mock_run_interactive.assert_not_called()
    mock_run_query.assert_called_once()
    called_query = mock_run_query.call_args[0][0]
    assert called_query == "profile TP53 in AML"


def test_no_args_enters_interactive_mode():
    with patch("ct.cli.run_query") as mock_run_query, patch(
        "ct.cli.run_interactive"
    ) as mock_run_interactive:
        result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    mock_run_query.assert_not_called()
    mock_run_interactive.assert_called_once()


def test_prompt_api_key_uses_masked_prompt_and_shows_hint():
    from ct.cli import _prompt_api_key

    with patch("prompt_toolkit.prompt", return_value="sk-ant-test") as mock_prompt, patch(
        "ct.cli.console.print"
    ) as mock_print:
        result = _prompt_api_key()

    assert result == "sk-ant-test"
    mock_prompt.assert_called_once_with(
        "  Enter your Anthropic API key: ",
        is_password=True,
    )
    mock_print.assert_any_call("  [dim]Pasted/typed characters will appear as *[/dim]")


def test_entry_routes_plain_invocation_to_hidden_run(monkeypatch):
    called = {}

    def fake_app(*, args, prog_name):
        called["args"] = args
        called["prog_name"] = prog_name

    monkeypatch.setattr("ct.cli.app", fake_app)
    monkeypatch.setattr("sys.argv", ["ct", "profile", "TP53"])

    from ct.cli import entry

    entry()

    assert called["prog_name"] == "ct"
    assert called["args"] == ["run", "profile", "TP53"]


def test_entry_preserves_explicit_subcommand(monkeypatch):
    called = {}

    def fake_app(*, args, prog_name):
        called["args"] = args
        called["prog_name"] = prog_name

    monkeypatch.setattr("ct.cli.app", fake_app)
    monkeypatch.setattr("sys.argv", ["ct", "config", "show"])

    from ct.cli import entry

    entry()

    assert called["prog_name"] == "ct"
    assert called["args"] == ["config", "show"]


def test_entry_preserves_trace_subcommand(monkeypatch):
    called = {}

    def fake_app(*, args, prog_name):
        called["args"] = args
        called["prog_name"] = prog_name

    monkeypatch.setattr("ct.cli.app", fake_app)
    monkeypatch.setattr("sys.argv", ["ct", "trace", "diagnose"])

    from ct.cli import entry

    entry()

    assert called["prog_name"] == "ct"
    assert called["args"] == ["trace", "diagnose"]


def test_config_set_agent_profile_applies_preset():
    cfg = Config(data={})
    with patch("ct.agent.config.Config.load", return_value=cfg), patch.object(
        cfg, "save"
    ) as mock_save:
        result = runner.invoke(app, ["config", "set", "agent.profile", "enterprise"])

    assert result.exit_code == 0
    assert "applied preset settings" in result.stdout
    assert cfg.get("agent.profile") == "enterprise"
    assert cfg.get("agent.quality_gate_strict") is True
    mock_save.assert_called_once()


def test_config_set_agent_profile_rejects_invalid_value():
    cfg = Config(data={})
    with patch("ct.agent.config.Config.load", return_value=cfg), patch.object(
        cfg, "save"
    ) as mock_save:
        result = runner.invoke(app, ["config", "set", "agent.profile", "invalid"])

    assert result.exit_code == 2
    assert "Invalid agent.profile" in result.stdout
    mock_save.assert_not_called()


def test_knowledge_status_command():
    fake_summary = {
        "path": "/tmp/substrate.json",
        "schema_version": 1,
        "n_entities": 3,
        "n_relations": 2,
        "n_evidence": 5,
        "entity_types": {"gene": 2, "disease": 1},
    }
    with patch("ct.kb.substrate.KnowledgeSubstrate") as mock_cls:
        mock_cls.return_value.summary.return_value = fake_summary
        result = runner.invoke(app, ["knowledge", "status"])
    assert result.exit_code == 0
    assert "Knowledge Substrate" in result.stdout
    assert "Entities" in result.stdout


def test_knowledge_ingest_error_exits_nonzero():
    with patch("ct.kb.ingest.KnowledgeIngestionPipeline") as mock_pipeline:
        mock_pipeline.return_value.ingest.return_value = {"error": "boom"}
        result = runner.invoke(app, ["knowledge", "ingest", "evidence_store"])
    assert result.exit_code == 2
    assert "boom" in result.stdout


def test_knowledge_benchmark_strict_failure_exits_nonzero():
    class FakeSuite:
        def run(self):
            return {
                "total_cases": 2,
                "expected_behavior_matches": 1,
                "pass_rate": 0.5,
            }

        def gate(self, summary, min_pass_rate=0.9):
            return {
                "ok": False,
                "message": "failed",
            }

    with patch("ct.kb.benchmarks.BenchmarkSuite.load", return_value=FakeSuite()):
        result = runner.invoke(app, ["knowledge", "benchmark", "--strict"])
    assert result.exit_code == 2


def test_trace_diagnose_command_outputs_summary(tmp_path):
    trace = TraceLogger("cli-trace-ok")
    trace.query_start("q1")
    trace.plan([], query="q1")
    trace.step_start(1, "files.create_file", {"path": "a.txt"})
    trace.step_complete(1, "files.create_file", {"summary": "ok"}, duration_s=0.1)
    trace.synthesize_start()
    trace.synthesize_end(token_count=10, duration_s=0.1)
    trace.query_end(iterations=1, total_steps=1, completed_steps=1, failed_steps=0)

    path = tmp_path / "cli-trace-ok.trace.jsonl"
    trace.save(path)

    result = runner.invoke(app, ["trace", "diagnose", "--path", str(path)])
    assert result.exit_code == 0
    assert "Trace Diagnostics" in result.stdout
    assert "Queries" in result.stdout
    assert "Step fails" in result.stdout


def test_trace_diagnose_strict_exits_on_unclosed_query(tmp_path):
    trace = TraceLogger("cli-trace-bad")
    trace.query_start("unfinished")
    trace.plan([], query="unfinished")
    # Intentionally omit query_end

    path = tmp_path / "cli-trace-bad.trace.jsonl"
    trace.save(path)

    result = runner.invoke(app, ["trace", "diagnose", "--path", str(path), "--strict"])
    assert result.exit_code == 2


def test_trace_export_creates_bundle(tmp_path):
    trace = TraceLogger("cli-export")
    trace.query_start("q1")
    trace.plan([], query="q1")
    trace.step_start(1, "files.create_file", {"path": "a.txt"})
    trace.step_complete(1, "files.create_file", {"summary": "ok"}, duration_s=0.1)
    trace.synthesize_start()
    trace.synthesize_end(token_count=10, duration_s=0.1)
    trace.query_end(iterations=1, total_steps=1, completed_steps=1, failed_steps=0)

    trace_path = tmp_path / "cli-export.trace.jsonl"
    trace.save(trace_path)
    report_path = tmp_path / "report.md"
    report_path.write_text("# report", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "trace",
            "export",
            "--path",
            str(trace_path),
            "--report",
            str(report_path),
            "--out-dir",
            str(tmp_path / "exports"),
            "--no-zip",
        ],
    )
    assert result.exit_code == 0
    assert "Bundle exported:" in result.stdout

    bundles = sorted((tmp_path / "exports").glob("ct_run_bundle_*"))
    assert bundles
    bundle = bundles[-1]
    assert (bundle / "trace.jsonl").exists()
    assert (bundle / "trace_diagnostics.json").exists()
    assert (bundle / "query_summaries.json").exists()
    assert (bundle / "manifest.json").exists()
    assert (bundle / "report.md").exists()


def test_release_check_passes_with_no_tests_no_trace():
    cfg = Config(data={"llm.api_key": "x"})

    class FakeSuite:
        def run(self):
            return {
                "total_cases": 2,
                "expected_behavior_matches": 2,
                "pass_rate": 1.0,
            }

        def gate(self, summary, min_pass_rate=0.9):
            del summary, min_pass_rate
            return {
                "ok": True,
                "message": "passed",
            }

    with patch("ct.agent.config.Config.load", return_value=cfg), patch(
        "ct.agent.doctor.run_checks", return_value=[]
    ), patch("ct.agent.doctor.has_errors", return_value=False), patch(
        "ct.agent.doctor.to_table", return_value="doctor ok"
    ), patch("ct.kb.benchmarks.BenchmarkSuite.load", return_value=FakeSuite()):
        result = runner.invoke(app, ["release-check", "--no-tests", "--no-trace"])

    assert result.exit_code == 0
    assert "Release check passed" in result.stdout


def test_release_check_fails_when_pytest_step_fails():
    cfg = Config(data={"llm.api_key": "x"})

    class FakeSuite:
        def run(self):
            return {
                "total_cases": 2,
                "expected_behavior_matches": 2,
                "pass_rate": 1.0,
            }

        def gate(self, summary, min_pass_rate=0.9):
            del summary, min_pass_rate
            return {
                "ok": True,
                "message": "passed",
            }

    fail_proc = subprocess.CompletedProcess(args=["pytest"], returncode=1, stdout="boom", stderr="")
    with patch("ct.agent.config.Config.load", return_value=cfg), patch(
        "ct.agent.doctor.run_checks", return_value=[]
    ), patch("ct.agent.doctor.has_errors", return_value=False), patch(
        "ct.agent.doctor.to_table", return_value="doctor ok"
    ), patch("ct.kb.benchmarks.BenchmarkSuite.load", return_value=FakeSuite()), patch(
        "ct.cli.subprocess.run", return_value=fail_proc
    ):
        result = runner.invoke(app, ["release-check", "--no-trace"])

    assert result.exit_code == 2
    assert "Release check failed" in result.stdout


def test_release_check_fails_on_trace_integrity_issues(tmp_path):
    cfg = Config(data={"llm.api_key": "x"})

    class FakeSuite:
        def run(self):
            return {
                "total_cases": 2,
                "expected_behavior_matches": 2,
                "pass_rate": 1.0,
            }

        def gate(self, summary, min_pass_rate=0.9):
            del summary, min_pass_rate
            return {
                "ok": True,
                "message": "passed",
            }

    trace = TraceLogger("bad-trace")
    trace.query_start("unfinished query")
    trace.plan([], query="unfinished query")
    trace_path = tmp_path / "bad-trace.trace.jsonl"
    trace.save(trace_path)

    with patch("ct.agent.config.Config.load", return_value=cfg), patch(
        "ct.agent.doctor.run_checks", return_value=[]
    ), patch("ct.agent.doctor.has_errors", return_value=False), patch(
        "ct.agent.doctor.to_table", return_value="doctor ok"
    ), patch("ct.kb.benchmarks.BenchmarkSuite.load", return_value=FakeSuite()):
        result = runner.invoke(
            app,
            ["release-check", "--no-tests", "--trace-path", str(trace_path)],
        )

    assert result.exit_code == 2
    assert "Trace diagnostics detected integrity issues" in result.stdout


def test_release_check_pharma_policy_fails_without_profile():
    cfg = Config(data={"llm.api_key": "x", "agent.profile": "research"})
    with patch("ct.agent.config.Config.load", return_value=cfg), patch(
        "ct.agent.doctor.run_checks", return_value=[]
    ), patch("ct.agent.doctor.has_errors", return_value=False), patch(
        "ct.agent.doctor.to_table", return_value="doctor ok"
    ):
        result = runner.invoke(
            app,
            ["release-check", "--no-tests", "--no-benchmark", "--no-trace", "--pharma"],
        )

    assert result.exit_code == 2
    assert "Profile mismatch" in result.stdout


def test_release_check_pharma_policy_passes():
    cfg = Config(
        data={
            "llm.api_key": "x",
            "agent.profile": "pharma",
            "agent.synthesis_style": "pharma",
            "agent.quality_gate_strict": True,
            "agent.enable_experimental_tools": False,
            "agent.enable_claude_code_tool": False,
        }
    )
    with patch("ct.agent.config.Config.load", return_value=cfg), patch(
        "ct.agent.doctor.run_checks", return_value=[]
    ), patch("ct.agent.doctor.has_errors", return_value=False), patch(
        "ct.agent.doctor.to_table", return_value="doctor ok"
    ):
        result = runner.invoke(
            app,
            ["release-check", "--no-tests", "--no-benchmark", "--no-trace", "--pharma"],
        )

    assert result.exit_code == 0
    assert "Release check passed" in result.stdout
