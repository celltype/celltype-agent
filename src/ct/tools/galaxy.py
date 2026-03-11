"""
Galaxy tool discovery: search and explore 1,770+ bioinformatics tools from Galaxy.

Provides local keyword search against a bundled tool catalog, with optional
live API queries to usegalaxy.org for real-time details.
"""

import json
from pathlib import Path

from ct.tools import registry
from ct.tools.http_client import request_json


_CATALOG_PATH = Path(__file__).parent / "data" / "galaxy_catalog.json"
_catalog_cache = None


def _load_catalog() -> list[dict]:
    """Load Galaxy tool catalog from bundled JSON."""
    global _catalog_cache
    if _catalog_cache is None:
        with open(_CATALOG_PATH) as f:
            _catalog_cache = json.load(f)
    return _catalog_cache


def _score_tool(tool: dict, terms: list[str]) -> float:
    """Score a tool's relevance to search terms."""
    score = 0.0
    name_lower = tool.get("name", "").lower()
    desc_lower = tool.get("description", "").lower()
    section_lower = tool.get("section", "").lower()
    edam_str = " ".join(tool.get("edam_topics", []) + tool.get("edam_operations", [])).lower()

    for term in terms:
        t = term.lower()
        if t in name_lower:
            score += 10.0
        if t in desc_lower:
            score += 3.0
        if t in section_lower:
            score += 5.0
        if t in edam_str:
            score += 4.0
    return score


@registry.register(
    name="galaxy.tool_search",
    description="Search Galaxy's 1,770+ bioinformatics tools by keyword, returning matching tools with EDAM annotations",
    category="galaxy",
    parameters={
        "query": "Search keywords (e.g. 'variant calling', 'RNA-seq alignment', 'metagenomics')",
        "max_results": "Maximum tools to return (default 20)",
    },
    requires_data=[],
    usage_guide="You want to find bioinformatics tools available in Galaxy for a specific analysis task. "
                "Returns tool names, descriptions, EDAM topic/operation annotations, and input/output types. "
                "Use to recommend upstream/downstream analysis tools in workflows.",
)
def tool_search(query: str, max_results: int = 20, **kwargs) -> dict:
    """Search bundled Galaxy tool catalog by keyword."""
    query = (query or "").strip()
    if not query:
        return {"error": "Query is required", "summary": "No query provided for Galaxy tool search"}

    max_results = max(1, min(int(max_results or 20), 100))
    catalog = _load_catalog()
    terms = query.lower().split()

    scored = []
    for tool in catalog:
        s = _score_tool(tool, terms)
        if s > 0:
            scored.append((s, tool))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_results]

    results = []
    for score, tool in top:
        results.append({
            "tool_id": tool.get("id", ""),
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "section": tool.get("section", ""),
            "edam_topics": tool.get("edam_topics", []),
            "edam_operations": tool.get("edam_operations", []),
            "inputs": [inp.get("type", "") for inp in tool.get("inputs", [])],
            "outputs": [out.get("type", "") for out in tool.get("outputs", [])],
            "version": tool.get("version", ""),
            "relevance_score": round(score, 1),
        })

    if results:
        top_names = ", ".join(r["name"] for r in results[:5])
        summary = f"Galaxy tool search '{query}': {len(results)} matches. Top: {top_names}"
    else:
        summary = f"Galaxy tool search '{query}': no matching tools found"

    return {
        "summary": summary,
        "query": query,
        "n_results": len(results),
        "results": results,
    }


@registry.register(
    name="galaxy.tool_details",
    description="Get detailed information about a specific Galaxy tool including inputs, outputs, and description",
    category="galaxy",
    parameters={
        "tool_id": "Galaxy tool ID (e.g. 'toolshed.g2.bx.psu.edu/repos/devteam/fastqc/fastqc/0.74+galaxy0')",
        "tool_name": "Tool name to search for if tool_id not known (e.g. 'FastQC')",
        "live": "Query usegalaxy.org API for real-time data (default False, uses bundled catalog)",
    },
    requires_data=[],
    usage_guide="You want detailed information about a specific Galaxy tool — its inputs, outputs, parameters, and description. "
                "Use after galaxy.tool_search to get full details on a specific tool.",
)
def tool_details(tool_id: str = None, tool_name: str = None, live: bool = False, **kwargs) -> dict:
    """Get detailed info about a specific Galaxy tool."""
    if not tool_id and not tool_name:
        return {"error": "Either tool_id or tool_name is required", "summary": "No tool identifier provided"}

    # Live API query
    if live and tool_id:
        encoded_id = tool_id.replace("/", "%2F")
        url = f"https://usegalaxy.org/api/tools/{encoded_id}"
        data, error = request_json("GET", url, timeout=15, retries=1)
        if error:
            return {
                "error": f"Galaxy API request failed: {error}",
                "summary": f"Failed to fetch live details for {tool_id}: {error}",
            }
        if not isinstance(data, dict):
            return {"error": "Unexpected API response format", "summary": f"Galaxy API returned unexpected format for {tool_id}"}

        inputs = []
        for inp in data.get("inputs", []):
            if isinstance(inp, dict):
                inputs.append({
                    "name": inp.get("name", ""),
                    "label": inp.get("label", ""),
                    "type": inp.get("type", ""),
                    "optional": inp.get("optional", False),
                })

        outputs = []
        for out in data.get("outputs", []):
            if isinstance(out, dict):
                outputs.append({
                    "name": out.get("name", ""),
                    "format": out.get("format", ""),
                    "label": out.get("label", ""),
                })

        name = data.get("name", "")
        description = data.get("description", "")
        version = data.get("version", "")
        edam_topics = data.get("edam_topics", [])
        edam_operations = data.get("edam_operations", [])

        summary = f"{name} (v{version}): {description}"

        return {
            "summary": summary,
            "tool_id": tool_id,
            "name": name,
            "description": description,
            "version": version,
            "edam_topics": edam_topics,
            "edam_operations": edam_operations,
            "inputs": inputs,
            "outputs": outputs,
            "source": "usegalaxy.org API",
            "found": True,
        }

    # Search bundled catalog
    catalog = _load_catalog()
    match = None

    if tool_id:
        for tool in catalog:
            if tool.get("id", "") == tool_id:
                match = tool
                break

    if not match and tool_name:
        name_lower = tool_name.lower()
        for tool in catalog:
            if tool.get("name", "").lower() == name_lower:
                match = tool
                break
        # Partial match fallback
        if not match:
            for tool in catalog:
                if name_lower in tool.get("name", "").lower():
                    match = tool
                    break

    if not match:
        identifier = tool_id or tool_name
        return {
            "summary": f"Galaxy tool '{identifier}' not found in bundled catalog. Try live=True for API lookup.",
            "found": False,
        }

    inputs = []
    for inp in match.get("inputs", []):
        inputs.append({
            "name": inp.get("name", ""),
            "type": inp.get("type", ""),
        })

    outputs = []
    for out in match.get("outputs", []):
        outputs.append({
            "name": out.get("name", ""),
            "type": out.get("type", ""),
        })

    name = match.get("name", "")
    description = match.get("description", "")
    version = match.get("version", "")

    summary = f"{name} (v{version}): {description}"

    return {
        "summary": summary,
        "tool_id": match.get("id", ""),
        "name": name,
        "description": description,
        "version": version,
        "section": match.get("section", ""),
        "edam_topics": match.get("edam_topics", []),
        "edam_operations": match.get("edam_operations", []),
        "inputs": inputs,
        "outputs": outputs,
        "tool_shed_url": match.get("tool_shed_url", ""),
        "source": "bundled catalog",
        "found": True,
    }
