# celltype-agent — Project Instructions

## What This Is

celltype-agent is an autonomous agent for drug discovery research — like Claude Code, but for biology.
It takes natural language questions about compounds/targets/indications and executes
multi-step research workflows using computational biology tools.

## Architecture

**Claude Agent SDK agentic loop**: Query → Claude (plans, calls tools, self-corrects, synthesizes) → Report

Uses ClaudeSDKClient with an in-process MCP server exposing all domain tools. Claude orchestrates
the full research workflow within a single agentic session (up to 30 tool-use turns).

### Key directories:
```
src/ct/
├── agent/          # Runner, MCP server, system prompt, config
├── tools/          # All research tools (190+), registered via @registry.register()
├── data/           # Data loaders (DepMap, PRISM, L1000, proteomics)
├── models/         # LLM client abstraction
└── ui/             # Interactive terminal
```

## Tool Pattern

Every tool follows this exact pattern:
```python
@registry.register(
    name="category.tool_name",
    description="What this tool does",
    category="category",
    parameters={"param": "description"},
    requires_data=["proteomics"],  # optional
)
def tool_name(param: str = "default", **kwargs) -> dict:
    """Docstring."""
    # ... implementation ...
    return {
        "summary": "Human-readable result summary",
        # ... additional data fields ...
    }
```

Rules:
- Name prefix MUST match category
- Always accept `**kwargs`
- Always return a dict with a `"summary"` key
- Use lazy imports for data loaders inside the function body
- Use `from ct.data.loaders import load_X` pattern

## Commands

```bash
ct --version                    # Check version
ct "your question"              # Single query
ct                              # Interactive mode
ct tool list                    # List all tools
ct config set key value         # Set config
ct data pull depmap             # Download dataset
pytest tests/ -v                # Run tests
pip install -e ".[dev]"         # Install for development
```

## Testing

Tests use mocked data loaders — never require real datasets.
Mock pattern: `@patch("ct.tools.module.load_X")` to inject test DataFrames.
