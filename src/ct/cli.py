"""
ct CLI entry point.

Usage:
    ct                              # Interactive mode
    ct "your question"              # Single query
    ct --smiles "CCO" "Profile"     # With compound context
    ct config set key value         # Configuration
    ct data pull depmap             # Data management
"""

import os
import json
import shutil
import subprocess
import sys
import typer
from typing import Optional
from pathlib import Path
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ct import __version__
from ct.agent.session import Session
from ct.ui.terminal import InteractiveTerminal


# ─── Startup banner ─────────────────────────────────────────
BANNER = """
[bold #50fa7b] ██████╗███████╗██╗     ██╗    ████████╗██╗   ██╗██████╗ ███████╗[/]
[bold #40f695]██╔════╝██╔════╝██║     ██║    ╚══██╔══╝╚██╗ ██╔╝██╔══██╗██╔════╝[/]
[bold #30f1b0]██║     █████╗  ██║     ██║       ██║    ╚████╔╝ ██████╔╝█████╗  [/]
[bold #20edca]██║     ██╔══╝  ██║     ██║       ██║     ╚██╔╝  ██╔═══╝ ██╔══╝  [/]
[bold #10e9e4]╚██████╗███████╗███████╗███████╗  ██║      ██║   ██║     ███████╗[/]
[bold #00e5ff] ╚═════╝╚══════╝╚══════╝╚══════╝  ╚═╝      ╚═╝   ╚═╝     ╚══════╝[/]
"""

app = typer.Typer(
    name="ct",
    help=(
        "CellType CLI — An autonomous agent for drug discovery research.\n\n"
        "Common usage:\n"
        '  ct "your research question"\n'
        '  ct --smiles "CCO" "Profile this compound"\n'
        "  ct config show\n"
        "  ct tool list"
    ),
    no_args_is_help=False,
)
console = Console()

# ─── Config subcommand ────────────────────────────────────────

config_app = typer.Typer(help="Manage ct configuration")
app.add_typer(config_app, name="config")


@config_app.command("set")
def config_set(key: str, value: str):
    """Set a configuration value."""
    from ct.agent.config import Config
    cfg = Config.load()
    try:
        cfg.set(key, value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)
    cfg.save()
    if key == "agent.profile":
        console.print(
            f"  [green]Set[/green] {key} = {cfg.get('agent.profile')} "
            "(applied preset settings)"
        )
    else:
        console.print(f"  [green]Set[/green] {key} = {value}")


@config_app.command("get")
def config_get(key: str):
    """Get a configuration value."""
    from ct.agent.config import Config
    cfg = Config.load()
    val = cfg.get(key)
    console.print(f"  {key} = {val}")


@config_app.command("show")
def config_show():
    """Show all configuration."""
    from ct.agent.config import Config
    cfg = Config.load()
    console.print(cfg.to_table())


@config_app.command("validate")
def config_validate():
    """Validate configuration and report issues."""
    from ct.agent.config import Config
    cfg = Config.load()
    issues = cfg.validate()
    if not issues:
        console.print("[green]Configuration is valid. No issues found.[/green]")
        return
    console.print(f"[yellow]Found {len(issues)} issue(s):[/yellow]")
    for issue in issues:
        console.print(f"  - {issue}")
    raise typer.Exit(code=2)


# ─── Keys command ────────────────────────────────────────────

@app.command("keys")
def keys_cmd():
    """Show status of optional API keys and what they unlock."""
    from ct.agent.config import Config
    cfg = Config.load()
    console.print(cfg.keys_table())


@app.command("setup")
def setup_cmd(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Anthropic API key (non-interactive mode)"),
):
    """Interactive setup wizard — configure ct for first use."""
    from ct.agent.config import Config

    cfg = Config.load()

    # Azure AI Foundry: skip interactive key prompt when Foundry is configured
    if os.environ.get("ANTHROPIC_FOUNDRY_API_KEY") or os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE"):
        console.print("\n  [green]Azure AI Foundry detected. No API key needed.[/green]")
        cfg.set("llm.provider", "anthropic")
        cfg.save()
        return

    console.print()
    console.print(
        Panel(
            "[bold]Welcome to CellType[/bold]\n\n"
            "This wizard will configure ct for first use.\n"
            "You need an Anthropic API key to get started.",
            title="[cyan]ct setup[/cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # Determine the key — non-interactive flag, existing config, env var, or prompt
    existing_key = cfg.llm_api_key()

    if api_key:
        # Non-interactive mode
        chosen_key = api_key
    elif existing_key:
        masked = existing_key[:7] + "..." + existing_key[-4:] if len(existing_key) > 11 else "***"
        console.print(f"  API key already configured: [green]{masked}[/green]")
        try:
            keep = input("  Keep existing key? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Setup cancelled.[/dim]")
            raise typer.Exit()
        if keep in ("", "y", "yes"):
            chosen_key = existing_key
            console.print("  [green]Keeping existing key.[/green]")
        else:
            chosen_key = _prompt_api_key()
    else:
        # Check env var
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        if env_key:
            masked = env_key[:7] + "..." + env_key[-4:] if len(env_key) > 11 else "***"
            console.print(f"  Found ANTHROPIC_API_KEY in environment: [green]{masked}[/green]")
            try:
                save_it = input("  Save to ct config? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print("\n  [dim]Setup cancelled.[/dim]")
                raise typer.Exit()
            if save_it in ("", "y", "yes"):
                chosen_key = env_key
            else:
                chosen_key = _prompt_api_key()
        else:
            chosen_key = _prompt_api_key()

    # Validate key format
    if not chosen_key or not chosen_key.startswith("sk-ant-"):
        console.print(
            "\n  [yellow]Warning:[/yellow] Key doesn't start with 'sk-ant-'. "
            "Anthropic API keys typically begin with 'sk-ant-api03-'."
        )
        try:
            proceed = input("  Continue anyway? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Setup cancelled.[/dim]")
            raise typer.Exit()
        if proceed not in ("y", "yes"):
            console.print("  [dim]Setup cancelled.[/dim]")
            raise typer.Exit()

    # Save
    cfg.set("llm.api_key", chosen_key)
    cfg.set("llm.provider", "anthropic")
    cfg.save()
    console.print("\n  [green]API key saved to ~/.ct/config.json[/green]")

    # Quick validation (full diagnostics available via `ct doctor`)
    console.print()
    console.print("  [cyan]Validating API key...[/cyan]")
    from ct.agent.doctor import run_checks, has_errors
    checks = run_checks(cfg)
    if has_errors(checks):
        console.print("  [red]✗[/red] Issues detected. Run [cyan]ct doctor[/cyan] for details.")
    else:
        console.print("  [green]✓ API key valid[/green]")

    # Cloud setup
    _setup_cloud(cfg)

    # Done
    console.print()
    console.print(
        Panel(
            "[bold green]You're all set![/bold green]\n\n"
            "  [cyan]ct[/cyan]                      Interactive mode\n"
            '  [cyan]ct "your question"[/cyan]      Single query\n'
            "  [cyan]ct doctor[/cyan]               Full health check\n"
            "  [cyan]ct keys[/cyan]                 Optional API keys\n"
            "  [cyan]ct setup-gpu[/cyan]            Switch to local GPU",
            title="[green]Quick Start[/green]",
            border_style="green",
        )
    )


def _setup_cloud(cfg):
    """Cloud-first setup: set cloud mode and log user in."""
    console.print()
    console.print(
        Panel(
            "Some tools like protein structure prediction\n"
            "and molecular docking require GPUs. We can\n"
            "run these jobs for you in the cloud.\n\n"
            "New accounts include [bold]$10 in free\n"
            "credits[/bold] to get started.\n\n"
            "Prefer your own GPU? Switch anytime\n"
            "with [cyan]ct setup-gpu[/cyan].",
            title="[cyan]CellType Cloud[/cyan]",
            border_style="cyan",
        )
    )

    cfg.set("compute.mode", "cloud")
    cfg.set("gpu.setup_completed", True)
    cfg.save()
    console.print()

    from ct.cloud.auth import is_logged_in
    if not is_logged_in():
        login_cmd()
    else:
        from ct.cloud.auth import get_user_email
        email = get_user_email() or "unknown"
        console.print(f"  Already logged in as [bold]{email}[/bold].")


def _setup_gpu(cfg):
    """GPU compute setup wizard (cloud vs local choice).

    Flow:
      1. Ask cloud vs local
      2. (local) Detect GPU, check VRAM, show per-tool compatibility
      3. (no GPU) Offer cloud fallback
    """
    from ct.cloud.router import _detect_local_gpu_info, get_gpu_tool_compatibility, _check_docker
    import ct.cloud.router as _router_mod

    console.print()
    console.print(
        Panel(
            "[bold]GPU Compute Setup[/bold]\n\n"
            "Some tools (AlphaFold, ESMFold, DiffDock) need GPUs for\n"
            "structure prediction and molecular docking.\n\n"
            "  [cyan]CellType Cloud[/cyan]  — serverless GPUs, pay-per-use, no setup\n"
            "  [cyan]Local GPU[/cyan]       — use your own NVIDIA GPU via Docker",
            title="[cyan]GPU Setup[/cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # ── Step 1: Cloud or local? ──
    try:
        choice = input("  Use CellType Cloud for GPU tasks? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [dim]Setup cancelled.[/dim]")
        raise typer.Exit()

    if choice in ("", "y", "yes"):
        cfg.set("compute.mode", "cloud")
        cfg.set("gpu.setup_completed", True)
        cfg.save()
        console.print("  [green]compute.mode = cloud[/green]")
        console.print()

        from ct.cloud.auth import is_logged_in
        if not is_logged_in():
            login_cmd()
        else:
            from ct.cloud.auth import get_user_email
            email = get_user_email() or "unknown"
            console.print(f"  Already logged in as [bold]{email}[/bold].")
        return

    # ── Step 2: Local — detect GPU ──
    console.print()
    console.print("  [cyan]Detecting local GPU...[/cyan]")
    _router_mod._gpu_info_cache = None  # Force fresh detection

    gpus = _detect_local_gpu_info()

    if not gpus:
        # ── Step 3: No GPU ──
        console.print("  [yellow]No NVIDIA GPU detected.[/yellow]")
        console.print()
        console.print(
            "  GPU-accelerated tools (AlphaFold, ESMFold, DiffDock) will not\n"
            "  be available. Most structural biology analysis will fail."
        )
        console.print()

        try:
            fallback = input(
                "  Would you like to use CellType Cloud instead? [Y/n] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Setup cancelled.[/dim]")
            raise typer.Exit()

        if fallback in ("", "y", "yes"):
            cfg.set("compute.mode", "cloud")
            cfg.set("gpu.setup_completed", True)
            cfg.save()
            console.print("  [green]compute.mode = cloud[/green]\n")
            from ct.cloud.auth import is_logged_in
            if not is_logged_in():
                login_cmd()
            return
        else:
            cfg.set("compute.mode", "local")
            cfg.set("gpu.setup_completed", True)
            cfg.save()
            console.print(
                "  [yellow]compute.mode = local[/yellow] (GPU tools will be unavailable)"
            )
            return

    # ── GPU found ──
    best_gpu = max(gpus, key=lambda g: g.vram_mb)
    console.print(
        f"  Found: [bold green]{best_gpu.name}[/bold green] "
        f"({best_gpu.vram_gb}GB VRAM)"
    )

    cfg.set("gpu.name", best_gpu.name)
    cfg.set("gpu.vram_mb", str(best_gpu.vram_mb))

    # Check per-tool compatibility
    compat = get_gpu_tool_compatibility(gpus)
    compatible = [c for c in compat if c["compatible"]]
    incompatible = [c for c in compat if not c["compatible"]]

    console.print()
    if compatible:
        console.print("  [green]Compatible tools:[/green]")
        for c in compatible:
            console.print(
                f"    [green]OK[/green]  {c['tool_name']} "
                f"(needs {c['min_vram_gb']}GB)"
            )

    if incompatible:
        console.print()
        console.print("  [yellow]Incompatible tools (need more VRAM):[/yellow]")
        for c in incompatible:
            console.print(
                f"    [red]--[/red]  {c['tool_name']} "
                f"(needs {c['min_vram_gb']}GB, you have {c['gpu_vram_gb']}GB)"
            )
        console.print()
        console.print(
            "  [dim]Incompatible tools will automatically run on CellType Cloud\n"
            "  if you're logged in, or be skipped otherwise.[/dim]"
        )

    if not incompatible:
        console.print()
        console.print("  [green]All GPU tools are compatible with your GPU.[/green]")

    # Check Docker
    console.print()
    docker_ok, docker_err = _check_docker()
    if docker_ok:
        console.print("  [green]Docker + NVIDIA Container Toolkit: OK[/green]")
    else:
        console.print(f"  [yellow]{docker_err}[/yellow]")

    cfg.set("compute.mode", "local")
    cfg.set("gpu.setup_completed", True)
    cfg.save()
    console.print()
    console.print("  [green]compute.mode = local[/green]")


@app.command("setup-gpu")
def setup_gpu_cmd():
    """Reconfigure GPU compute settings (cloud vs local)."""
    from ct.agent.config import Config
    cfg = Config.load()
    _setup_gpu(cfg)


def _prompt_api_key() -> str:
    """Prompt user for API key with masked input."""
    import getpass
    console.print("  Get your key at: [link=https://console.anthropic.com/settings/keys]console.anthropic.com/settings/keys[/link]")
    console.print()
    try:
        key = getpass.getpass("  Enter your Anthropic API key: ")
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [dim]Setup cancelled.[/dim]")
        raise typer.Exit()
    return key.strip()


@app.command("doctor")
def doctor_cmd():
    """Run environment and configuration health checks."""
    from ct.agent.config import Config
    from ct.agent.doctor import run_checks, to_table, has_errors

    cfg = Config.load()
    checks = run_checks(cfg, session=Session(config=cfg, mode="batch"))
    console.print(to_table(checks))

    if has_errors(checks):
        console.print(
            "\n[red]Blocking issues found.[/red] "
            "Fix errors above, then rerun `ct doctor`."
        )
        raise typer.Exit(code=1)

    console.print("\n[green]No blocking issues found.[/green]")


# ─── Data subcommand ──────────────────────────────────────────

data_app = typer.Typer(help="Manage local datasets")
app.add_typer(data_app, name="data")


@data_app.command("pull")
def data_pull(
    dataset: str = typer.Argument(help="Dataset to download (depmap, prism, msigdb, alphafold)"),
    output: Optional[Path] = typer.Option(None, help="Output directory"),
):
    """Download a dataset for local use."""
    from ct.data.downloader import download_dataset
    download_dataset(dataset, output)


@data_app.command("status")
def data_status():
    """Show status of local datasets."""
    from ct.data.downloader import dataset_status
    console.print(dataset_status())


# ─── Tool subcommands (direct tool access) ────────────────────

tool_app = typer.Typer(help="Run individual tools directly")
app.add_typer(tool_app, name="tool")


@tool_app.command("list")
def tool_list(
    gpu: bool = typer.Option(False, "--gpu", help="Show only GPU/high-memory tools with local compatibility"),
):
    """List all available tools."""
    from ct.tools import registry, ensure_loaded, tool_load_errors
    ensure_loaded()

    if gpu:
        # Show GPU tools with hardware compatibility
        from ct.cloud.router import _detect_local_gpu_info

        gpus = _detect_local_gpu_info()
        best_gpu = max(gpus, key=lambda g: g.vram_mb) if gpus else None

        table = Table(title="GPU/High-Memory Tools")
        table.add_column("Tool", style="cyan")
        table.add_column("Type")
        table.add_column("Requirement")
        table.add_column("Local", style="bold")
        table.add_column("Description")

        for tool in registry.list_tools():
            if not tool.requires_gpu and not tool.cpu_only:
                continue

            if tool.cpu_only:
                req = f"{tool.min_ram_gb}GB RAM"
                tool_type = "CPU"
                compat = "[yellow]cloud-only[/yellow]"
            else:
                req = f"{tool.min_vram_gb}GB VRAM"
                tool_type = "GPU"
                if best_gpu and best_gpu.vram_gb >= tool.min_vram_gb:
                    compat = "[green]compatible[/green]"
                elif best_gpu:
                    compat = f"[red]need {tool.min_vram_gb}GB (have {best_gpu.vram_gb}GB)[/red]"
                else:
                    compat = "[red]no GPU[/red]"

            table.add_row(tool.name, tool_type, req, compat, tool.description[:60])

        console.print(table)
        if best_gpu:
            console.print(f"\nLocal GPU: {best_gpu.name} ({best_gpu.vram_gb}GB VRAM)")
        else:
            console.print("\n[yellow]No local GPU detected.[/yellow]")
    else:
        console.print(registry.list_tools_table())

    errors = tool_load_errors()
    if errors:
        names = ", ".join(sorted(errors.keys())[:8])
        extra = "" if len(errors) <= 8 else f" (+{len(errors) - 8} more)"
        console.print(
            f"[yellow]Warning:[/yellow] {len(errors)} tool module(s) failed to load: "
            f"{names}{extra}"
        )


@tool_app.command("pull")
def tool_pull(
    tool_name: str = typer.Argument(..., help="Tool name (e.g., structure.esmfold)"),
    force: bool = typer.Option(False, "--force", help="Force rebuild even if image exists"),
):
    """Build Docker image and pre-download model weights for a GPU tool.

    This prepares a tool for local GPU execution:
      1. Builds the Docker image from docker-images/<tool>/
      2. Runs the container once to download model weights to ~/.cache/

    After pulling, the tool runs locally without any network access.
    """
    import subprocess
    import yaml
    from pathlib import Path

    # Find tool directory by scanning src/ct/tools/*/tool.yaml
    tools_dir = Path(__file__).parent / "tools"
    docker_dir = None
    config = None

    for candidate in sorted(tools_dir.iterdir()):
        yaml_path = candidate / "tool.yaml"
        if candidate.is_dir() and yaml_path.exists():
            try:
                with open(yaml_path) as f:
                    cfg = yaml.safe_load(f)
                if cfg.get("name") == tool_name:
                    config = cfg
                    docker_dir = candidate
                    break
            except Exception:
                continue

    if not config or not docker_dir:
        console.print(f"[red]Tool '{tool_name}' not found. Available tools:[/red]")
        for d in sorted(tools_dir.iterdir()):
            yp = d / "tool.yaml"
            if d.is_dir() and yp.exists():
                try:
                    with open(yp) as f:
                        n = yaml.safe_load(f).get("name", "")
                    console.print(f"  {n}")
                except Exception:
                    pass
        raise typer.Exit(code=1)

    display_name = config.get("display_name", tool_name)
    docker_image = config.get("docker_image", "")
    if not docker_image:
        console.print(f"[red]No docker_image configured for {tool_name}.[/red]")
        raise typer.Exit(code=1)

    # Step 1: Build Docker image
    if not force:
        # Check if image already exists
        check = subprocess.run(["docker", "image", "inspect", docker_image],
                               capture_output=True, timeout=10)
        if check.returncode == 0:
            console.print(f"  Image {docker_image} already exists (use --force to rebuild)")
        else:
            force = True  # Image doesn't exist, must build

    if force:
        console.print(f"[bold]Building {docker_image}...[/bold]")
        # Copy tool_entrypoint.py if missing
        entrypoint = docker_dir / "tool_entrypoint.py"
        if not entrypoint.exists():
            src = tools_dir / "tool_entrypoint.py"
            if src.exists():
                import shutil
                shutil.copy2(src, entrypoint)

        result = subprocess.run(
            ["docker", "build", "-t", docker_image, str(docker_dir)],
            timeout=3600,
        )
        if result.returncode != 0:
            console.print(f"[red]Build failed for {docker_image}[/red]")
            raise typer.Exit(code=1)
        console.print(f"[green]Built {docker_image}[/green]")

    # Step 2: Pre-download weights by running container with cache mounts
    # This triggers the first-run weight download inside the container
    compute = config.get("compute", {})
    cpu_only = compute.get("cpu_only", False) or not compute.get("requires_gpu", True)
    if cpu_only:
        console.print(f"[green]{display_name} is CPU-only, no weights to download.[/green]")
        return

    console.print(f"[bold]Pre-downloading weights for {display_name}...[/bold]")

    from ct.cloud.local_runner import LocalRunner
    runner = LocalRunner()
    cache_mounts = runner._get_cache_mounts()

    # Build a minimal test run to trigger weight downloads
    import tempfile, json
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Minimal input — just enough to trigger model loading
        test_input = {"sequence": "MKWV"}  # tiny sequence
        if "diffdock" in tool_name:
            test_input = {"protein_pdb": "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00\nEND",
                          "ligand_smiles": "C", "num_poses": 1}
        elif "proteinmpnn" in tool_name:
            test_input = {"backbone_pdb": "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\nATOM      2  CA  ALA A   1       1.5   0.0   0.0  1.00  0.00\nATOM      3  C   ALA A   1       2.5   1.2   0.0  1.00  0.00\nATOM      4  O   ALA A   1       2.2   2.4   0.0  1.00  0.00\nEND",
                          "num_sequences": 1}
        elif "rfdiffusion" in tool_name:
            test_input = {"target_pdb": "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00\nEND",
                          "num_designs": 1}
        elif "evo2" in tool_name and "protein" in tool_name:
            test_input = {"target_function": "test"}
        elif "evo2" in tool_name:
            test_input = {"dna_sequence": "ATG"}
        elif "genmol" in tool_name:
            test_input = {"scaffold_smiles": "C", "num_molecules": 1}
        elif "molmim" in tool_name:
            test_input = {"input_smiles": "C", "num_variants": 1}
        elif "msa" in tool_name:
            test_input = {"sequence": "MKWV"}
        elif "multimer" in tool_name:
            test_input = {"sequences": ["MK", "WV"]}

        (tmpdir / "input.json").write_text(json.dumps(test_input))

        gpu_flags = ["--gpus", "all"] if not cpu_only else []
        cmd = ["docker", "run", "--rm"] + gpu_flags + [
            "-v", f"{tmpdir}:/workspace",
        ] + cache_mounts + [
            "-e", "INPUT_FILE=/workspace/input.json",
            "-e", "OUTPUT_FILE=/workspace/output.json",
            docker_image,
        ]

        result = subprocess.run(cmd, timeout=1800)

        # Container ran — weights are downloaded to cache regardless of
        # inference errors (which are expected with minimal test inputs)
        if result.returncode == 0:
            console.print(f"[green]{display_name} ready. Weights cached.[/green]")
        else:
            # Even on non-zero exit, weights may have been partially downloaded.
            # Check if the container at least started (weights download happens early).
            console.print(f"[green]{display_name} image built. Weights will download on first real use.[/green]")


@tool_app.command("build")
def tool_build(
    tool_name: str = typer.Argument(..., help="Tool name (e.g., structure.esmfold)"),
):
    """Build a Docker image for a GPU tool from the manifest."""
    from ct.cloud.manifest import get_tool_config
    from ct.cloud.image_builder import generate_dockerfile

    config = get_tool_config(tool_name)
    if not config:
        console.print(f"[red]Tool '{tool_name}' not found in manifest.[/red]")
        raise typer.Exit(code=1)

    display_name = config.get("display_name", tool_name)
    docker_image = config.get("docker_image", f"celltype/{tool_name.split('.')[-1]}:latest")

    console.print(f"[bold]Building Docker image for {display_name}...[/bold]")

    # Generate Dockerfile
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        tmpdir = Path(tmpdir)
        dockerfile_content = generate_dockerfile(config)
        (tmpdir / "Dockerfile").write_text(dockerfile_content)

        # Copy entrypoint
        entrypoint_src = Path(__file__).parent / "cloud" / "tool_entrypoint.py"
        if entrypoint_src.exists():
            import shutil
            shutil.copy2(entrypoint_src, tmpdir / "tool_entrypoint.py")

        # Create placeholder implementation
        impl_key = tool_name.split(".")[-1]
        (tmpdir / "implementation.py").write_text(
            f'"""Placeholder implementation for {display_name}."""\n\n'
            f'def run(**kwargs):\n'
            f'    return {{"summary": "{display_name} placeholder"}}\n'
        )

        # Build
        import subprocess
        result = subprocess.run(
            ["docker", "build", "-t", docker_image, "."],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            console.print(f"[red]Build failed:[/red]\n{result.stderr[:500]}")
            raise typer.Exit(code=1)

        console.print(f"[green]Built {docker_image}[/green]")


@tool_app.command("setup")
def tool_setup(
    tool_name: str = typer.Argument(..., help="Tool name (e.g., structure.esmfold)"),
):
    """Pull Docker image and download model weights for a GPU tool."""
    from ct.cloud.manifest import get_tool_config

    config = get_tool_config(tool_name)
    if not config:
        console.print(f"[red]Tool '{tool_name}' not found in manifest.[/red]")
        raise typer.Exit(code=1)

    display_name = config.get("display_name", tool_name)
    docker_image = config.get("docker_image", "")

    console.print(f"[bold]Setting up {display_name}...[/bold]")

    # Step 1: Pull Docker image
    if docker_image:
        console.print(f"\n[bold]Step 1: Pulling Docker image {docker_image}...[/bold]")
        import subprocess
        result = subprocess.run(
            ["docker", "pull", docker_image],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            console.print(f"[yellow]Warning: Could not pull {docker_image}. You may need to build it first.[/yellow]")
        else:
            console.print(f"[green]Pulled {docker_image}[/green]")

    # Step 2: Download weights
    console.print(f"\n[bold]Step 2: Downloading model weights...[/bold]")
    from ct.cloud.weight_downloader import pull_tool_weights
    result = pull_tool_weights(tool_name)

    if "error" in result:
        console.print(f"[red]{result['summary']}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]{result['summary']}[/green]")
    console.print(f"\n[bold green]{display_name} is ready![/bold green]")


# ─── Knowledge subcommands ────────────────────────────────────

knowledge_app = typer.Typer(help="Manage knowledge substrate, ingestion, and quality gates")
app.add_typer(knowledge_app, name="knowledge")


# ─── Trace subcommands ────────────────────────────────────────

trace_app = typer.Typer(help="Inspect and diagnose execution traces")
app.add_typer(trace_app, name="trace")


def _latest_trace_path() -> Optional[Path]:
    from ct.agent.trace import TraceLogger

    traces_dir = TraceLogger.traces_dir()
    traces = list(traces_dir.glob("*.trace.jsonl"))
    if not traces:
        return None
    return max(traces, key=lambda p: p.stat().st_mtime)


def _resolve_trace_path(path: Optional[Path], session_id: Optional[str]) -> Optional[Path]:
    from ct.agent.trace import TraceLogger

    if path is not None and session_id is not None:
        console.print("[red]Use either --path or --session-id, not both.[/red]")
        raise typer.Exit(code=2)

    if path is not None:
        return path
    if session_id:
        return TraceLogger.traces_dir() / f"{session_id}.trace.jsonl"
    return _latest_trace_path()


def _latest_report_path(output_base: Optional[str] = None) -> Optional[Path]:
    reports_dir = (
        Path(output_base) / "reports"
        if output_base
        else Path.cwd() / "outputs" / "reports"
    )
    if not reports_dir.exists():
        return None
    reports = list(reports_dir.glob("*.md"))
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime)


def _trace_has_issues(diag: dict) -> bool:
    return any(
        (
            diag.get("unclosed_queries"),
            diag.get("queries_with_no_plan"),
            diag.get("queries_with_no_completion"),
            diag.get("queries_with_synthesis_mismatch"),
        )
    )


def _print_trace_diagnostics_table(diag: dict, title: str):
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Session", diag.get("session_id", "(unknown)") or "(unknown)")
    table.add_row("Events", str(diag.get("event_count", 0)))
    table.add_row("Queries", str(diag.get("query_count", 0)))
    table.add_row(
        "Query starts / ends",
        f"{diag.get('query_start_count', 0)} / {diag.get('query_end_count', 0)}",
    )
    table.add_row("Step starts", str(diag.get("total_step_start_count", 0)))
    table.add_row("Step completes", str(diag.get("total_step_complete_count", 0)))
    table.add_row("Step fails", str(diag.get("total_step_fail_count", 0)))
    table.add_row("Step retries", str(diag.get("total_step_retry_count", 0)))
    table.add_row("Unclosed queries", str(diag.get("unclosed_queries", [])))
    table.add_row("Queries with failures", str(diag.get("queries_with_failures", [])))
    table.add_row("Queries with no plan", str(diag.get("queries_with_no_plan", [])))
    table.add_row(
        "Queries with no completion",
        str(diag.get("queries_with_no_completion", [])),
    )
    table.add_row(
        "Synthesis mismatches",
        str(diag.get("queries_with_synthesis_mismatch", [])),
    )
    console.print(table)


def _run_step_command(label: str, cmd: list[str], env: Optional[dict] = None) -> bool:
    console.print(f"\n[bold cyan]{label}[/bold cyan]")
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if stdout:
        console.print(stdout)
    if stderr:
        style = "yellow" if proc.returncode == 0 else "red"
        console.print(stderr, style=style)
    if proc.returncode == 0:
        console.print(f"[green]PASS[/green] {label}")
        return True
    console.print(f"[red]FAIL[/red] {label} (exit={proc.returncode})")
    return False


@trace_app.command("diagnose")
def trace_diagnose(
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Path to a trace JSONL file"),
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s", help="Session ID (looks up ~/.ct/traces/<id>.trace.jsonl)"),
    as_json: bool = typer.Option(False, "--json", help="Print diagnostics as JSON"),
    show_queries: bool = typer.Option(False, "--show-queries", help="Show per-query diagnostics table"),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if health issues are detected"),
):
    """Diagnose trace health (query integrity, failures, synthesis lifecycle)."""
    from ct.agent.trace import TraceLogger

    trace_path = _resolve_trace_path(path, session_id)
    if trace_path is None:
        console.print("[yellow]No trace files found in ~/.ct/traces[/yellow]")
        raise typer.Exit(code=2)
    if not trace_path.exists():
        console.print(f"[red]Trace file not found:[/red] {trace_path}")
        raise typer.Exit(code=2)

    trace = TraceLogger.load(trace_path)
    diag = trace.diagnostics()

    if as_json:
        console.print_json(data=diag)
    else:
        _print_trace_diagnostics_table(diag, title=f"Trace Diagnostics: {trace_path.name}")

        if show_queries:
            q_table = Table(title="Per-Query Diagnostics")
            q_table.add_column("#", style="cyan")
            q_table.add_column("Closed")
            q_table.add_column("Plans")
            q_table.add_column("Step OK")
            q_table.add_column("Step Fail")
            q_table.add_column("Retries")
            q_table.add_column("Synth start/end")
            q_table.add_column("Query")
            for q in diag["queries"]:
                q_table.add_row(
                    str(q["query_number"]),
                    "yes" if q["closed"] else "no",
                    str(q["plan_count"]),
                    str(q["step_complete_count"]),
                    str(q["step_fail_count"]),
                    str(q["step_retry_count"]),
                    f"{q['synthesize_start_count']}/{q['synthesize_end_count']}",
                    (q["query"] or "")[:80],
                )
            console.print(q_table)

    has_issues = _trace_has_issues(diag)
    if strict and has_issues:
        raise typer.Exit(code=2)


@trace_app.command("export")
def trace_export(
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Path to a trace JSONL file"),
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s", help="Session ID (looks up ~/.ct/traces/<id>.trace.jsonl)"),
    report: Optional[Path] = typer.Option(None, "--report", "-r", help="Optional markdown report to include"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Bundle output directory (default: ~/.ct/exports)"),
    zip_bundle: bool = typer.Option(True, "--zip/--no-zip", help="Also produce a zip archive"),
):
    """Export a reproducible run bundle (trace, diagnostics, report, metadata)."""
    from ct.agent.config import Config
    from ct.agent.trace import TraceLogger
    from ct.agent.trajectory import Trajectory

    trace_path = _resolve_trace_path(path, session_id)
    if trace_path is None:
        console.print("[yellow]No trace files found in ~/.ct/traces[/yellow]")
        raise typer.Exit(code=2)
    if not trace_path.exists():
        console.print(f"[red]Trace file not found:[/red] {trace_path}")
        raise typer.Exit(code=2)

    trace = TraceLogger.load(trace_path)
    diag = trace.diagnostics()

    cfg = Config.load()
    resolved_report = report
    if resolved_report is None:
        resolved_report = _latest_report_path(cfg.get("sandbox.output_dir"))
    if resolved_report is not None and not resolved_report.exists():
        console.print(f"[red]Report file not found:[/red] {resolved_report}")
        raise typer.Exit(code=2)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out_dir or (Path.home() / ".ct" / "exports")
    bundle_dir = base / f"ct_run_bundle_{trace.session_id or 'session'}_{ts}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    trace_copy = bundle_dir / "trace.jsonl"
    shutil.copy2(trace_path, trace_copy)
    (bundle_dir / "trace.txt").write_text(trace.to_text(), encoding="utf-8")
    (bundle_dir / "trace_diagnostics.json").write_text(
        json.dumps(diag, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (bundle_dir / "query_summaries.json").write_text(
        json.dumps(trace.query_summaries(), indent=2),
        encoding="utf-8",
    )

    copied_report = None
    if resolved_report is not None:
        copied_report = bundle_dir / "report.md"
        shutil.copy2(resolved_report, copied_report)

    copied_session = None
    session_file = None
    if trace.session_id:
        session_file = Trajectory.sessions_dir() / f"{trace.session_id}.jsonl"
    if session_file is not None and session_file.exists():
        copied_session = bundle_dir / "session.jsonl"
        shutil.copy2(session_file, copied_session)

    manifest = {
        "generated_at_utc": ts,
        "session_id": trace.session_id,
        "source_trace": str(trace_path),
        "included_files": {
            "trace_jsonl": str(trace_copy),
            "trace_txt": str(bundle_dir / "trace.txt"),
            "trace_diagnostics_json": str(bundle_dir / "trace_diagnostics.json"),
            "query_summaries_json": str(bundle_dir / "query_summaries.json"),
            "report_md": str(copied_report) if copied_report else None,
            "session_jsonl": str(copied_session) if copied_session else None,
        },
        "note": (
            "If report was auto-selected, it is the latest markdown report by mtime "
            "from sandbox.output_dir/reports."
            if report is None
            else "Report path explicitly provided."
        ),
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    console.print(f"[green]Bundle exported:[/green] {bundle_dir}")
    if copied_report:
        console.print(f"[dim]Included report:[/dim] {resolved_report}")
    else:
        console.print("[yellow]No report included (none found/provided).[/yellow]")

    if zip_bundle:
        archive = shutil.make_archive(str(bundle_dir), "zip", root_dir=bundle_dir)
        console.print(f"[green]Zip archive:[/green] {archive}")


@app.command("release-check")
def release_check_cmd(
    run_tests: bool = typer.Option(True, "--tests/--no-tests", help="Run local pytest regression suite"),
    run_benchmark: bool = typer.Option(True, "--benchmark/--no-benchmark", help="Run strict knowledge benchmark gate"),
    run_trace: bool = typer.Option(True, "--trace/--no-trace", help="Run strict diagnostics on latest trace"),
    trace_path: Optional[Path] = typer.Option(None, "--trace-path", help="Trace path for diagnostics"),
    trace_required: bool = typer.Option(False, "--trace-required", help="Fail if no trace file is found"),
    include_live: bool = typer.Option(False, "--live", help="Also run live API smoke + live E2E prompt matrix"),
    matrix_limit: int = typer.Option(10, "--matrix-limit", help="Prompt limit for live E2E matrix"),
    matrix_strict: bool = typer.Option(True, "--matrix-strict/--no-matrix-strict", help="Enable strict assertions in live E2E matrix"),
    matrix_max_failed: int = typer.Option(1, "--matrix-max-failed", help="Max failed prompts allowed in strict matrix mode"),
    require_profile: Optional[str] = typer.Option(None, "--require-profile", help="Require agent.profile to match (e.g. pharma)"),
    pharma: bool = typer.Option(False, "--pharma", help="Enforce pharma deployment policy checks"),
):
    """Run a production release gate: doctor + tests + benchmark + trace diagnostics."""
    from ct.agent.config import Config
    from ct.agent.doctor import has_errors, run_checks, to_table
    from ct.agent.trace import TraceLogger
    from ct.kb.benchmarks import BenchmarkSuite

    failed = False

    console.print("\n[bold]Release Check[/bold]")

    cfg = Config.load()
    if pharma and not require_profile:
        require_profile = "pharma"

    if require_profile:
        expected = require_profile.strip().lower()
        actual = str(cfg.get("agent.profile", "research")).strip().lower()
        if actual != expected:
            console.print(
                f"[red]Profile mismatch:[/red] expected '{expected}', got '{actual}'."
            )
            failed = True

    if pharma:
        policy_issues = []
        if str(cfg.get("agent.synthesis_style", "standard")).strip().lower() != "pharma":
            policy_issues.append("agent.synthesis_style must be 'pharma'")
        if not bool(cfg.get("agent.quality_gate_strict", False)):
            policy_issues.append("agent.quality_gate_strict must be true")
        if bool(cfg.get("agent.enable_experimental_tools", False)):
            policy_issues.append("agent.enable_experimental_tools must be false")
        if bool(cfg.get("agent.enable_claude_code_tool", False)):
            policy_issues.append("agent.enable_claude_code_tool must be false")
        if policy_issues:
            console.print("[red]Pharma policy checks failed:[/red]")
            for issue in policy_issues:
                console.print(f"- {issue}")
            failed = True

    checks = run_checks(cfg)
    console.print(to_table(checks))
    if has_errors(checks):
        console.print("[red]Doctor checks have blocking errors.[/red]")
        failed = True

    if run_tests:
        ok = _run_step_command(
            "Local test suite",
            ["pytest", "-q", "tests", "-m", "not api_smoke and not e2e and not e2e_matrix"],
        )
        failed = failed or (not ok)

    if run_benchmark:
        suite = BenchmarkSuite.load()
        summary = suite.run()
        gate = suite.gate(summary, min_pass_rate=0.9)

        table = Table(title="Release Benchmark Gate")
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        table.add_row("Total cases", str(summary["total_cases"]))
        table.add_row("Expected behavior matches", str(summary["expected_behavior_matches"]))
        table.add_row("Pass rate", str(summary["pass_rate"]))
        table.add_row("Gate", gate["message"])
        console.print(table)

        if not gate["ok"]:
            console.print("[red]Benchmark release gate failed.[/red]")
            failed = True

    if run_trace:
        resolved_trace = trace_path or _latest_trace_path()
        if resolved_trace is None or not resolved_trace.exists():
            msg = "No trace file found for diagnostics."
            if trace_required:
                console.print(f"[red]{msg}[/red]")
                failed = True
            else:
                console.print(f"[yellow]{msg}[/yellow]")
        else:
            trace = TraceLogger.load(resolved_trace)
            diag = trace.diagnostics()
            _print_trace_diagnostics_table(diag, title=f"Trace Diagnostics: {resolved_trace.name}")
            if _trace_has_issues(diag):
                console.print("[red]Trace diagnostics detected integrity issues.[/red]")
                failed = True

    if include_live:
        smoke_env = dict(os.environ)
        smoke_env["CT_RUN_API_SMOKE"] = "1"
        smoke_env.setdefault("CT_API_SMOKE_STRICT", "1")
        smoke_ok = _run_step_command(
            "Live API smoke checks",
            ["pytest", "-q", "tests/test_api_smoke.py"],
            env=smoke_env,
        )
        failed = failed or (not smoke_ok)

        matrix_env = dict(os.environ)
        matrix_env["CT_RUN_E2E_MATRIX"] = "1"
        matrix_env["CT_E2E_MATRIX_LIMIT"] = str(max(1, matrix_limit))
        matrix_env["CT_E2E_MATRIX_STRICT"] = "1" if matrix_strict else "0"
        matrix_env["CT_E2E_MATRIX_MAX_FAILED_QUERIES"] = str(max(0, matrix_max_failed))
        matrix_ok = _run_step_command(
            "Live E2E prompt matrix",
            ["pytest", "-q", "tests/test_e2e_matrix.py", "--run-e2e"],
            env=matrix_env,
        )
        failed = failed or (not matrix_ok)

    if failed:
        console.print("\n[red]Release check failed.[/red]")
        raise typer.Exit(code=2)

    console.print("\n[green]Release check passed.[/green]")


@knowledge_app.command("status")
def knowledge_status():
    """Show knowledge substrate status."""
    from ct.kb.substrate import KnowledgeSubstrate

    substrate = KnowledgeSubstrate()
    summary = substrate.summary()
    table = Table(title="Knowledge Substrate")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Path", summary["path"])
    table.add_row("Schema Version", str(summary["schema_version"]))
    table.add_row("Entities", str(summary["n_entities"]))
    table.add_row("Relations", str(summary["n_relations"]))
    table.add_row("Evidence", str(summary["n_evidence"]))
    for et, count in sorted(summary.get("entity_types", {}).items()):
        table.add_row(f"entity_type:{et}", str(count))
    console.print(table)


@knowledge_app.command("ingest")
def knowledge_ingest(
    source: str = typer.Argument(..., help="Source: evidence_store | pubmed | openalex | opentargets"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Query for API sources"),
    max_results: int = typer.Option(10, "--max-results", help="Max records for API sources"),
    scan_limit: int = typer.Option(1000, "--scan-limit", help="Max local evidence rows to scan"),
):
    """Ingest knowledge into canonical substrate."""
    from ct.kb.ingest import KnowledgeIngestionPipeline

    pipeline = KnowledgeIngestionPipeline()
    result = pipeline.ingest(
        source=source,
        query=query,
        max_results=max_results,
        scan_limit=scan_limit,
    )
    if result.get("error"):
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(code=2)
    console.print(result.get("summary", "Ingestion completed."))


@knowledge_app.command("search")
def knowledge_search(
    query: str = typer.Argument(..., help="Search text"),
    limit: int = typer.Option(20, "--limit", help="Maximum entities to return"),
):
    """Search canonical entities."""
    from ct.kb.substrate import KnowledgeSubstrate

    substrate = KnowledgeSubstrate()
    entities = substrate.search_entities(query, limit=limit)
    table = Table(title=f"Knowledge Search: {query}")
    table.add_column("Entity ID", style="cyan")
    table.add_column("Type")
    table.add_column("Name")
    table.add_column("Synonyms", style="dim")
    for entity in entities:
        table.add_row(entity.id, entity.entity_type, entity.name, ", ".join(entity.synonyms[:4]))
    console.print(table)


@knowledge_app.command("related")
def knowledge_related(
    entity_id: str = typer.Argument(..., help="Canonical entity id (e.g., gene:TP53)"),
    predicate: Optional[str] = typer.Option(None, "--predicate", help="Filter predicate"),
    limit: int = typer.Option(20, "--limit", help="Maximum relations"),
):
    """Show related entities for an entity."""
    from ct.kb.substrate import KnowledgeSubstrate

    substrate = KnowledgeSubstrate()
    rows = substrate.related_entities(entity_id, predicate=predicate, limit=limit)
    table = Table(title=f"Related Entities: {entity_id}")
    table.add_column("Predicate", style="cyan")
    table.add_column("Other Entity")
    table.add_column("Support")
    table.add_column("Contradict")
    table.add_column("Avg Score")
    for row in rows:
        table.add_row(
            row["predicate"],
            row["other_entity_id"],
            str(row["support_claims"]),
            str(row["contradict_claims"]),
            str(row["average_claim_score"]),
        )
    console.print(table)


@knowledge_app.command("rank")
def knowledge_rank(
    entity_id: Optional[str] = typer.Option(None, "--entity-id", help="Entity id filter"),
    predicate: Optional[str] = typer.Option(None, "--predicate", help="Predicate filter"),
    limit: int = typer.Option(20, "--limit", help="Maximum relations"),
):
    """Rank relations by evidence strength."""
    from ct.kb.reasoning import EvidenceReasoner
    from ct.kb.substrate import KnowledgeSubstrate

    reasoner = EvidenceReasoner(KnowledgeSubstrate())
    rows = reasoner.rank_relations(entity_id=entity_id, predicate=predicate, limit=limit)
    table = Table(title="Ranked Relations")
    table.add_column("Relation", style="cyan")
    table.add_column("Score")
    table.add_column("Claims")
    for row in rows:
        relation = f"{row['subject_id']} --{row['predicate']}--> {row['object_id']}"
        table.add_row(relation, str(row["score"]), str(row["n_claims"]))
    console.print(table)


@knowledge_app.command("contradictions")
def knowledge_contradictions(
    entity_id: Optional[str] = typer.Option(None, "--entity-id", help="Entity id filter"),
    predicate: Optional[str] = typer.Option(None, "--predicate", help="Predicate filter"),
):
    """Detect contradictory evidence clusters."""
    from ct.kb.reasoning import EvidenceReasoner
    from ct.kb.substrate import KnowledgeSubstrate

    reasoner = EvidenceReasoner(KnowledgeSubstrate())
    rows = reasoner.detect_contradictions(entity_id=entity_id, predicate=predicate)
    table = Table(title="Contradictions")
    table.add_column("Relation", style="cyan")
    table.add_column("Support")
    table.add_column("Contradict")
    table.add_column("Support Score")
    table.add_column("Contradict Score")
    for row in rows:
        relation = f"{row['subject_id']} --{row['predicate']}--> {row['object_id']}"
        table.add_row(
            relation,
            str(row["support_claims"]),
            str(row["contradict_claims"]),
            str(row["support_score"]),
            str(row["contradict_score"]),
        )
    console.print(table)


@knowledge_app.command("schema-check")
def knowledge_schema_check():
    """Run schema drift checks against external integration baselines."""
    from ct.kb.schema_monitor import SchemaMonitor

    monitor = SchemaMonitor()
    results = monitor.check()
    summary = monitor.summarize(results)
    table = Table(title="Schema Drift Monitor")
    table.add_column("Monitor", style="cyan")
    table.add_column("Status")
    table.add_column("Added")
    table.add_column("Removed")
    table.add_column("Error")
    for row in summary["results"]:
        table.add_row(
            row["monitor"],
            row["status"],
            str(len(row["added_paths"])),
            str(len(row["removed_paths"])),
            row.get("error", ""),
        )
    console.print(table)
    if summary["counts"].get("drift", 0) > 0 or summary["counts"].get("error", 0) > 0:
        raise typer.Exit(code=2)


@knowledge_app.command("schema-update")
def knowledge_schema_update(monitor: Optional[str] = typer.Option(None, "--monitor", help="Single monitor to update")):
    """Update schema drift baselines from current responses."""
    from ct.kb.schema_monitor import SchemaMonitor

    mon = SchemaMonitor()
    results = mon.update_baseline(monitor=monitor)
    summary = mon.summarize(results)
    console.print(f"Updated schema baseline for {summary['total']} monitor(s).")


@knowledge_app.command("benchmark")
def knowledge_benchmark(
    min_pass_rate: float = typer.Option(0.9, "--min-pass-rate", help="Release gate threshold"),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if gate fails"),
):
    """Run benchmark suite and evaluate release gate."""
    from ct.kb.benchmarks import BenchmarkSuite

    suite = BenchmarkSuite.load()
    summary = suite.run()
    gate = suite.gate(summary, min_pass_rate=min_pass_rate)

    table = Table(title="Knowledge Benchmarks")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Total cases", str(summary["total_cases"]))
    table.add_row("Expected behavior matches", str(summary["expected_behavior_matches"]))
    table.add_row("Pass rate", str(summary["pass_rate"]))
    table.add_row("Gate", gate["message"])
    console.print(table)
    if strict and not gate["ok"]:
        raise typer.Exit(code=2)


# ─── Report subcommands ──────────────────────────────────────

report_app = typer.Typer(help="Generate and publish reports")
app.add_typer(report_app, name="report")


@report_app.command("list")
def report_list():
    """List available markdown reports."""
    from ct.agent.config import Config

    cfg = Config.load()
    reports_dir = (
        Path(cfg.get("sandbox.output_dir")) / "reports"
        if cfg.get("sandbox.output_dir")
        else Path.cwd() / "outputs" / "reports"
    )
    if not reports_dir.exists():
        console.print("[dim]No reports directory found.[/dim]")
        raise typer.Exit()

    reports = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        console.print("[dim]No reports found.[/dim]")
        raise typer.Exit()

    table = Table(title="Reports")
    table.add_column("#", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Size", style="dim")
    table.add_column("Modified")
    for i, r in enumerate(reports[:20], 1):
        size = r.stat().st_size
        size_str = f"{size / 1024:.1f}K" if size > 1024 else f"{size}B"
        mtime = datetime.fromtimestamp(r.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        table.add_row(str(i), r.name, size_str, mtime)
    console.print(table)


@report_app.command("publish")
def report_publish(
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Markdown report to convert"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output HTML path"),
):
    """Convert a markdown report to a shareable HTML page."""
    from ct.agent.config import Config
    from ct.reports.html import publish_report

    if path is None:
        cfg = Config.load()
        path = _latest_report_path(cfg.get("sandbox.output_dir"))
        if path is None:
            console.print("[yellow]No reports found. Run a query first.[/yellow]")
            raise typer.Exit(code=2)

    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(code=2)

    result = publish_report(path, out_path=out)
    console.print(f"[green]Published:[/green] {result}")


@report_app.command("show")
def report_show(
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="HTML report to open"),
):
    """Open an HTML report in the default browser."""
    import webbrowser

    from ct.agent.config import Config
    from ct.reports.html import publish_report

    if path is None:
        cfg = Config.load()
        md_path = _latest_report_path(cfg.get("sandbox.output_dir"))
        if md_path is None:
            console.print("[yellow]No reports found.[/yellow]")
            raise typer.Exit(code=2)
        html_path = md_path.with_suffix(".html")
        if not html_path.exists():
            html_path = publish_report(md_path)
            console.print(f"[dim]Auto-published: {html_path}[/dim]")
        path = html_path

    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(code=2)

    webbrowser.open(f"file://{path.resolve()}")
    console.print(f"[green]Opened in browser:[/green] {path}")


@report_app.command("notebook")
def report_notebook(
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID (prefix or full). Default: most recent"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output notebook path"),
    html: bool = typer.Option(False, "--html", help="Also export as HTML"),
):
    """Export an agent trace as a Jupyter notebook (.ipynb)."""
    import re
    from ct.agent.trace_store import TraceStore

    # Find trace file
    trace_path = TraceStore.find_trace(session)
    if trace_path is None:
        console.print("[yellow]No trace files found. Run a query first to generate a trace.[/yellow]")
        raise typer.Exit(code=2)

    console.print(f"  [dim]Trace:[/dim] {trace_path.name}")

    # Lazy import nbformat
    try:
        from ct.reports.notebook import trace_to_notebook, save_notebook
    except ImportError:
        console.print("[red]nbformat is required. Install with:[/red] pip install nbformat")
        raise typer.Exit(code=2)

    # Convert trace to notebook
    nb = trace_to_notebook(trace_path)

    # Determine output path
    if out is None:
        from ct.agent.config import Config
        cfg = Config.load()
        reports_dir = (
            Path(cfg.get("sandbox.output_dir")) / "reports"
            if cfg.get("sandbox.output_dir")
            else Path.cwd() / "outputs" / "reports"
        )
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", trace_path.stem.replace(".trace", "")).strip("_")
        out = reports_dir / f"{slug}.ipynb"

    out_path = save_notebook(nb, out)
    console.print(f"  [green]Notebook:[/green] {out_path}")

    # Optional HTML export
    if html:
        try:
            import nbconvert
            from nbconvert import HTMLExporter
            exporter = HTMLExporter()
            html_body, _ = exporter.from_notebook_node(nb)
            html_path = out_path.with_suffix(".html")
            html_path.write_text(html_body, encoding="utf-8")
            console.print(f"  [green]HTML:[/green] {html_path}")
        except ImportError:
            console.print(
                "[yellow]nbconvert not installed. Falling back to markdown-based HTML.[/yellow]\n"
                "  [dim]Install with: pip install nbconvert[/dim]"
            )
            # Fall back to existing HTML renderer on markdown cells
            from ct.reports.html import render_html_report
            md_parts = [c.source for c in nb.cells if c.cell_type == "markdown"]
            md_text = "\n\n".join(md_parts)
            html_content = render_html_report(md_text, title="ct Notebook Export")
            html_path = out_path.with_suffix(".html")
            html_path.write_text(html_content, encoding="utf-8")
            console.print(f"  [green]HTML (markdown only):[/green] {html_path}")


# ─── Case study subcommands ─────────────────────────────────

case_study_app = typer.Typer(help="Run curated drug case studies")
app.add_typer(case_study_app, name="case-study")


@case_study_app.command("list")
def case_study_list():
    """List available curated case studies."""
    from ct.agent.case_studies import CASE_STUDIES

    table = Table(title="Case Studies")
    table.add_column("ID", style="cyan")
    table.add_column("Drug")
    table.add_column("Threads", style="dim")
    table.add_column("Description")
    for case_id, case in CASE_STUDIES.items():
        table.add_row(
            case_id,
            case.name,
            str(len(case.thread_goals)),
            case.description[:80] + ("..." if len(case.description) > 80 else ""),
        )
    console.print(table)


@case_study_app.command("run")
def case_study_run(
    case_id: str = typer.Argument(..., help="Case study ID (e.g., revlimid, gleevec)"),
    threads: Optional[int] = typer.Option(None, "--threads", "-t", help="Number of parallel threads"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run a curated drug case study with multi-agent analysis."""
    from ct.agent.case_studies import CASE_STUDIES, run_case_study
    from ct.agent.config import Config
    from ct.reports.html import publish_report

    if case_id not in CASE_STUDIES:
        available = ", ".join(sorted(CASE_STUDIES.keys()))
        console.print(f"[red]Unknown case study '{case_id}'.[/red] Available: {available}")
        raise typer.Exit(code=2)

    cfg = Config.load()
    if model:
        cfg.set("llm.model", model)

    llm_issue = cfg.llm_preflight_issue()
    if llm_issue:
        console.print("\n  [yellow]First-time setup required.[/yellow]\n")
        setup_cmd(api_key=None)
        cfg = Config.load()
        if model:
            cfg.set("llm.model", model)
        llm_issue = cfg.llm_preflight_issue()
        if llm_issue:
            console.print(f"\n  [red]Setup incomplete:[/red] {llm_issue}")
            raise typer.Exit(code=2)

    session = Session(config=cfg, verbose=verbose)
    case = CASE_STUDIES[case_id]

    print_banner()
    console.print(Panel(
        f"[bold]{case.name}[/bold]\n[dim]{case.description}[/dim]",
        title="[cyan]Case Study[/cyan]",
        border_style="cyan",
    ))
    console.print()

    result = run_case_study(session, case_id, n_threads=threads)

    # Auto-publish HTML
    md_path = _latest_report_path(cfg.get("sandbox.output_dir"))
    if md_path:
        html_path = publish_report(md_path)
        console.print(f"\n  [green]HTML report:[/green] {html_path}")

    console.print()

# ─── Cloud / Auth commands ───────────────────────────────────────

@app.command("login")
def login_cmd():
    """Log in to CellType Cloud (required for GPU tools)."""
    from ct.cloud.auth import login, poll_for_approval, is_logged_in, get_user_email

    if is_logged_in():
        email = get_user_email() or "unknown"
        console.print(f"  Already logged in as [bold]{email}[/bold]. Run `ct logout` to switch accounts.")
        return

    try:
        result = login()
    except Exception as e:
        console.print(f"  [red]Could not reach CellType Cloud:[/red] {e}")
        raise typer.Exit(code=2)

    if result.get("already_logged_in"):
        console.print(f"  Already logged in as [bold]{result['email']}[/bold].")
        return

    session_code = result["session_code"]
    auth_url = result["auth_url"]

    console.print(f"\n  Create a free account or sign in (includes [bold]$10 free credits[/bold]):")
    console.print(f"  [link={auth_url}][cyan]{auth_url}[/cyan][/link]\n")

    # Auto-open browser on desktop platforms (skip SSH/headless Linux)
    import os, sys
    if sys.platform in ("darwin", "win32") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        import webbrowser
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

    console.print(f"  Confirmation code: [bold cyan]{session_code}[/bold cyan]\n")
    console.print("  Waiting for authorization...")

    try:
        auth = poll_for_approval(session_code)
        console.print(f"\n  [green]Logged in as {auth['email']}.[/green]")
    except RuntimeError as e:
        console.print(f"\n  [red]{e}[/red]")
        raise typer.Exit(code=2)


@app.command("logout")
def logout_cmd():
    """Log out of CellType Cloud."""
    from ct.cloud.auth import logout

    if logout():
        console.print("  Logged out.")
    else:
        console.print("  Not logged in.")


@app.command("account")
def account_cmd():
    """Show CellType Cloud account info."""
    from ct.cloud.auth import is_logged_in, get_user_email, get_token

    if not is_logged_in():
        console.print("  Not logged in. Run `ct login` to create a free account.")
        return

    email = get_user_email() or "unknown"
    console.print(f"  Email: [bold]{email}[/bold]")

    # Try to fetch balance from API
    try:
        token = get_token()
        if token:
            import httpx
            from ct.cloud.auth import _get_api_url

            with httpx.Client(timeout=15) as client:
                resp = client.get(
                    f"{_get_api_url()}/account/credits",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    balance = data.get("balance", 0.0)
                    console.print(f"  Balance: [bold]${balance:.2f}[/bold]")
                else:
                    console.print("  Balance: [dim]unavailable[/dim]")
    except Exception:
        console.print("  Balance: [dim]unavailable[/dim]")


@app.command("credits")
def credits_cmd():
    """Show CellType Cloud credit balance and recent usage."""
    from ct.cloud.auth import is_logged_in, get_token

    if not is_logged_in():
        console.print("  Not logged in. Run `ct login` to create a free account with starter credits.")
        return

    try:
        token = get_token()
        if not token:
            console.print("  Session expired. Run `ct login` to re-authenticate.")
            return

        import httpx
        from ct.cloud.auth import _get_api_url

        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{_get_api_url()}/account/credits",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                balance = data.get("balance", 0.0)
                console.print(f"  Balance: [bold]${balance:.2f}[/bold]")

                # Recent usage
                usage = data.get("recent_usage", [])
                if usage:
                    table = Table(title="Recent GPU Usage")
                    table.add_column("Date", style="dim")
                    table.add_column("Tool", style="cyan")
                    table.add_column("Cost", justify="right")
                    table.add_column("Duration", justify="right", style="dim")
                    for entry in usage[:10]:
                        table.add_row(
                            str(entry.get("date", ""))[:16],
                            str(entry.get("tool", "")),
                            f"${entry.get('cost', 0):.2f}",
                            f"{entry.get('duration_s', 0):.0f}s",
                        )
                    console.print(table)
                else:
                    console.print("  [dim]No recent GPU usage.[/dim]")

                if balance < 1.0:
                    console.print(
                        f"\n  [yellow]Low balance (${balance:.2f}).[/yellow] "
                        "Add credits at [link=https://celltype.com/billing]celltype.com/billing[/link]"
                    )
            else:
                console.print("  [red]Failed to fetch credits.[/red] Try again later.")
    except RuntimeError as e:
        console.print(f"  {e}")
    except Exception:
        console.print("  [red]CellType Cloud is temporarily unavailable.[/red]")


# ─── Main entry point ─────────────────────────────────────────

@app.command("run", hidden=True)
def run_cmd(
    query_parts: list[str] = typer.Argument(None, help="Research question to investigate"),
    smiles: Optional[str] = typer.Option(None, "--smiles", "-s", help="Compound SMILES string"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target protein (UniProt ID or gene symbol)"),
    indication: Optional[str] = typer.Option(None, "--indication", "-i", help="Cancer type / indication"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for reports"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    agents: Optional[int] = typer.Option(None, "--agents", "-a", help="Run with N parallel research agents"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Resume a previous session (ID or 'last')"),
    continue_last: bool = typer.Option(False, "--continue", "-c", help="Continue the most recent session"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    version: bool = typer.Option(False, "--version", "-V", help="Show version"),
):
    """
    CellType CLI — An autonomous agent for drug discovery research.

    Run without arguments for interactive mode.
    Pass a question for single-query mode.
    """
    if version:
        console.print(f"ct v{__version__}")
        raise typer.Exit()

    query = " ".join(query_parts).strip() if query_parts else None

    # Build context from flags
    context = {}
    if smiles:
        context["compound_smiles"] = smiles
    if target:
        context["target"] = target
    if indication:
        context["indication"] = indication

    # Determine session resume
    resume_id = None
    if continue_last:
        resume_id = "last"
    elif resume:
        resume_id = resume

    if query:
        # Single query mode
        run_query(query, context, output, model, verbose, agents=agents)
    else:
        # Interactive mode
        run_interactive(context, output, model, verbose, resume_id=resume_id)


def run_query(query: str, context: dict, output: Optional[Path],
              model: Optional[str], verbose: bool, agents: Optional[int] = None):
    """Execute a single research query."""
    from ct.agent.config import Config

    cfg = Config.load()
    if model:
        cfg.set("llm.model", model)

    llm_issue = cfg.llm_preflight_issue()
    if llm_issue:
        console.print("\n  [yellow]First-time setup required.[/yellow]\n")
        setup_cmd(api_key=None)
        cfg = Config.load()
        if model:
            cfg.set("llm.model", model)
        llm_issue = cfg.llm_preflight_issue()
        if llm_issue:
            console.print(f"\n  [red]Setup incomplete:[/red] {llm_issue}")
            raise typer.Exit(code=2)

    session = Session(config=cfg, verbose=verbose)

    print_banner()
    console.print(Panel(
        f"[bold]{query}[/bold]",
        title="[cyan]ct[/cyan]",
        border_style="cyan",
    ))
    console.print()

    # Multi-agent mode
    if agents is not None and agents > 1:
        from ct.agent.orchestrator import ResearchOrchestrator
        orchestrator = ResearchOrchestrator(session, n_threads=agents)
        result = orchestrator.run(query, context)

        if output:
            output.mkdir(parents=True, exist_ok=True)
            report_path = output / "report.md"
            report_path.write_text(result.to_markdown())
            console.print(f"\n  Report saved to {report_path}")

        console.print()
        return

    # Execute via Agent SDK runner (default) or legacy AgentLoop (fallback)
    use_sdk = cfg.get("agent.use_sdk", True)

    if use_sdk:
        from ct.agent.runner import AgentRunner
        agent = AgentRunner(session)
        result = agent.run(query, context)
    else:
        from ct.agent.loop import AgentLoop, ClarificationNeeded
        agent = AgentLoop(session)
        try:
            result = agent.run(query, context)
        except ClarificationNeeded as e:
            console.print(f"\n  [cyan]{e.clarification.question}[/cyan]")
            if e.clarification.suggestions:
                console.print(f"  [dim]e.g. {', '.join(e.clarification.suggestions[:3])}[/dim]")
            console.print(f"\n  [dim]Tip: provide context with --smiles, --target, or --indication flags.[/dim]")
            return

    # Output
    if output:
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "report.md"
        report_path.write_text(result.to_markdown())
        console.print(f"\n  Report saved to {report_path}")

    # Summary already streamed to stdout during synthesis
    console.print()


@app.command("bench")
def bench(
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Run a single question by ID"),
    parallel: int = typer.Option(10, "--parallel", "-p", help="Number of parallel workers"),
    timeout: int = typer.Option(300, "--timeout", help="Timeout per question in seconds"),
    max_turns: int = typer.Option(15, "--max-turns", help="Max agentic loop turns"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model override"),
    eval_model: str = typer.Option("claude-opus-4-6", "--eval-model", help="Model for LLM-as-judge evaluation"),
    manifest: str = typer.Option("/mnt/bixbench/manifest.json", "--manifest", help="Path to manifest JSON"),
    output: str = typer.Option("/mnt/bixbench/outputs", "--output", "-o", help="Output directory"),
    only_failed: bool = typer.Option(False, "--only-failed", help="Re-run only failed questions"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview questions without executing"),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip inline LLM evaluation"),
    force: bool = typer.Option(False, "--force", "-f", help="Clear previous results and re-run everything"),
    max_questions: Optional[int] = typer.Option(None, "--max-questions", "-n", help="Limit to first N questions"),
):
    """Run the BixBench-50 benchmark suite."""
    import shutil as _shutil
    from ct.bench.runner import BenchRunner

    if force:
        out = Path(output)
        for sub in ("results", "evals", ".preview_cache"):
            d = out / sub
            if d.exists():
                _shutil.rmtree(d)
        for f in ("all_results.json", "llm_eval.json"):
            p = out / f
            if p.exists():
                p.unlink()
        console.print(f"  [dim]Cleared {out}[/dim]")

    if dry_run:
        import json as _json
        with open(manifest) as f:
            questions = _json.load(f)
        if question:
            questions = [q for q in questions if q["question_id"] == question]
        if max_questions:
            questions = questions[:max_questions]

        table = Table(title=f"BixBench Dry Run — {len(questions)} questions")
        table.add_column("#", width=4)
        table.add_column("Question ID", style="cyan", width=14)
        table.add_column("Data", width=5)
        table.add_column("Question", max_width=60)
        table.add_column("Ideal", max_width=30)

        for i, q in enumerate(questions, 1):
            has_data = "Y" if q.get("data_dir") and Path(q["data_dir"]).exists() else "N"
            table.add_row(
                str(i), q["question_id"], has_data,
                q["question"][:60], q["ideal"][:30],
            )
        console.print(table)
        return

    runner = BenchRunner(
        manifest_path=manifest,
        output_dir=output,
        parallel=parallel,
        timeout=timeout,
        max_turns=max_turns,
        model=model,
        eval_model=eval_model,
        no_eval=no_eval,
        only_failed=only_failed,
        question_id=question,
        max_questions=max_questions,
    )

    summary = runner.run()
    if summary.get("total"):
        console.print(
            f"\n[bold]Score: {summary['passed']}/{summary['total']} "
            f"({summary['accuracy']:.1%})[/bold]"
        )


def print_banner():
    """Print the startup banner with molecule illustration."""
    from ct.tools import registry, ensure_loaded
    from rich.panel import Panel
    from rich.text import Text
    ensure_loaded()
    n_tools = len(registry.list_tools())
    
    # Print the ASCII logo (just the CELLTYPE art)
    console.print(BANNER) 
    
    # Create a nice enclosed dashboard panel for the metadata
    meta_text = Text.from_markup(
        f"[bold white]Autonomous Drug Discovery Agent[/]\n"
        f"[dim]v{__version__}  ·  {n_tools} tools loaded  ·  backed by[/dim] [bold white on #f26522] Y [/][bold #f26522] Combinator[/]",
        justify="center"
    )
    
    console.print(Panel(
        meta_text, 
        title="[bold cyan]CellType CLI[/]",
        border_style="dim",
        width=65
    ))


def run_interactive(context: dict, output: Optional[Path],
                    model: Optional[str], verbose: bool, resume_id: str = None):
    """Run interactive session."""
    from ct.agent.config import Config

    cfg = Config.load()
    if model:
        cfg.set("llm.model", model)

    llm_issue = cfg.llm_preflight_issue()
    if llm_issue:
        console.print("\n  [yellow]First-time setup required.[/yellow]\n")
        setup_cmd(api_key=None)
        # Reload config after setup
        cfg = Config.load()
        llm_issue = cfg.llm_preflight_issue()
        if llm_issue:
            console.print(f"\n  [red]Setup incomplete:[/red] {llm_issue}")
            return

    print_banner()

    # Show model info like Claude Code does
    console.print("  [dim]Type a research question, or /help for commands.[/dim]")
    console.print()

    terminal = InteractiveTerminal(config=cfg, verbose=verbose)
    terminal.run(initial_context=context, resume_id=resume_id)


def entry():
    """Package entry point."""
    from ct.update_checker import start_check, get_update_message
    start_check(__version__)

    # Check if stored token has been revoked (quick, non-blocking)
    try:
        from ct.cloud.auth import is_logged_in, check_auth
        if is_logged_in() and not check_auth():
            console.print("  [dim]Session revoked. You've been logged out.[/dim]")
    except Exception:
        pass

    argv = list(sys.argv[1:])
    passthrough = {
        "config",
        "data",
        "tool",
        "trace",
        "knowledge",
        "keys",
        "doctor",
        "setup",
        "release-check",
        "report",
        "case-study",
        "bench",
        "login",
        "logout",
        "account",
        "credits",
        "setup-gpu",
        "run",
        "--help",
        "-h",
        "--install-completion",
        "--show-completion",
    }

    # Route plain invocations to hidden `run` command so:
    #   ct                       -> interactive mode
    #   ct "question"            -> single-query mode
    #   ct --smiles ... "q"      -> single-query with context
    # while preserving explicit subcommands like `ct config ...`.
    if not argv or argv[0] not in passthrough:
        argv = ["run", *argv]

    # Show update notification if a newer version was found
    update_msg = get_update_message()
    if update_msg:
        console.print(f"\n  {update_msg}\n")

    app(args=argv, prog_name="ct")


if __name__ == "__main__":
    entry()
