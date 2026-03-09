"""
Unified system prompt builder for the Claude Agent SDK runner.

Merges the domain knowledge primer, workflow guides, bioinformatics code-gen
hints, synthesis rules, tool catalog, and dynamic data context into a single
system prompt that Claude uses throughout the agentic loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("ct.system_prompt")


# ---------------------------------------------------------------------------
# Identity / role preamble
# ---------------------------------------------------------------------------

_IDENTITY = """\
You are **celltype-cli**, an autonomous drug discovery research agent.

You have access to 190+ domain tools covering target discovery, chemistry,
expression, viability, safety, clinical development, omics, genomics, literature,
and more — plus a persistent Python sandbox (``run_python``) for custom analyses.

Your job: take a research question and answer it **completely**, using the right
tools and code, self-correcting as you go, and producing a publication-quality
synthesis at the end.

## Operating Mode
- You are in an agentic loop: call tools, see results, call more tools, then
  write your final answer as plain text (no tool call).
- Think step-by-step. Use tools to gather evidence, then synthesize.
- If a tool fails or returns unhelpful data, try a different approach or use
  your own knowledge to fill gaps.
- For data analysis questions, use ``run_python`` to load data, explore it,
  and compute the answer. Variables persist between calls.
"""


# ---------------------------------------------------------------------------
# Synthesis instructions (injected at the end)
# ---------------------------------------------------------------------------

_SYNTHESIS_INSTRUCTIONS = """\

## When You Are Ready to Answer

Write your final answer as a direct text response (do NOT call any more tools).
Your answer should be:

1. **Complete**: Address every part of the question. Decompose the question into
   sub-parts and make sure each is answered with specifics.
2. **Accurate**: Use tool results as primary evidence. Supplement with your
   domain knowledge. Never fabricate data.
3. **Data-rich**: Include specific gene names, cell lines, p-values, effect
   sizes, IC50 values, trial names, mutation positions, etc.
4. **Mechanistic**: Explain the biological *why*, not just the *what*.
5. **Actionable**: End with 3-5 specific experimental next steps (named assays,
   cell lines, concentrations, readouts).

BANNED PHRASES — never write these:
- "cannot be answered with the data retrieved"
- "failed to retrieve" / "failed to identify"
- "insufficient data" / "insufficient evidence"
- "No results were obtained"
If tools failed, pivot to answering from your knowledge instead.
"""

_STRUCTURE_INPUT_GUIDANCE = """\

## Structure File Inputs

For tools that accept `target_pdb`, `backbone_pdb`, or `protein_pdb`, you may
provide either:
- inline PDB/mmCIF text, or
- a local filesystem path to a `.pdb`, `.cif`, `.mmcif`, or `.ent` file.

If the structure is already saved locally, prefer passing the file path instead
of pasting the full structure text. The local runner and CellType Cloud client
will inline the file contents automatically before dispatch.
"""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    session,
    tool_names: list[str] | None = None,
    data_context: str | None = None,
    history: str | None = None,
) -> str:
    """Build the unified system prompt for the Agent SDK runner.

    Args:
        session: Active ct Session.
        tool_names: Names of tools available in the MCP server (for reference).
        data_context: Free-text description of available data files / directories.
        history: Prior conversation turns (for interactive multi-turn sessions).

    Returns:
        The complete system prompt string.
    """
    parts: list[str] = []

    # 1. Identity
    parts.append(_IDENTITY)

    # 2. Tool catalog (concise reference — full descriptions are in MCP tool defs)
    # NOTE: The Agent SDK exposes tool names+descriptions+schemas via MCP natively.
    # We only include a brief orientation here, NOT the full tool_descriptions_for_llm()
    # which would blow up the system prompt to 155K chars and hit ARG_MAX limits.
    if tool_names:
        parts.append(f"\n## Available Tools ({len(tool_names)} total)\n")
        parts.append(
            "You have access to all tools via MCP. Key tools:\n"
            "- **run_python**: Execute Python code in a sandbox (pd, np, plt, scipy, sklearn, pysam, gseapy, pydeseq2, BioPython). Variables persist between calls.\n"
            "- **run_r**: Execute R code directly. Prefer run_r over run_python for: natural splines (ns()), wilcox.test(), p.adjust(), fisher.test(), lm()/predict(), organism-specific KEGG ORA (use KEGGREST package: keggList, keggLink, keggGet for any organism code), and any analysis where R is the reference implementation. R and Python give DIFFERENT results for splines, multiple testing correction, and nonparametric tests — when the expected answer was computed in R, use R.\n"
            "- **literature.pubmed_search**, **literature.chembl_query**, **literature.openalex_search**: Literature/DB search\n"
            "- **data_api.opentargets_search**, **data_api.depmap_search**, **data_api.uniprot_lookup**: Platform APIs\n"
            "- **omics.geo_search**, **omics.geo_fetch**, **omics.deseq2**, **omics.dataset_info**: Omics data discovery + analysis\n"
            "- **expression.pathway_enrichment**, **expression.l1000_similarity**: Expression/pathway tools\n"
            "- **viability.dose_response**, **clinical.indication_map**, **clinical.trial_search**: Viability/clinical\n"
            "- **chemistry.descriptors**, **chemistry.sar_analyze**, **chemistry.pubchem_lookup**: Chemistry tools\n"
            "- **target.coessentiality**, **genomics.gwas_lookup**, **protein.function_predict**: Target/genomics tools\n"
            "- **safety.classify**, **safety.admet_predict**: Safety/ADMET tools\n"
            "\nFor data analysis questions, prefer **run_python** — it's the most powerful tool.\n"
            "For drug discovery questions, combine domain tools with your knowledge.\n"
        )

    # 3. Workflow guides (compact — key sequences for common tasks)
    try:
        from ct.agent.workflows import format_workflows_for_llm
        workflows = format_workflows_for_llm()
        if workflows:
            parts.append(workflows)
    except Exception as e:
        logger.warning("Could not load workflows: %s", e)

    # 4. Domain knowledge primer (CRITICAL for drug discovery accuracy)
    # NOTE: The KNOWLEDGE_PRIMER contains both tool orientation AND domain facts.
    # The tool orientation section overlaps with MCP tool descriptions but the
    # cross-disciplinary thinking patterns and domain-specific accuracy anchors
    # are essential. Include in full.
    try:
        from ct.agent.knowledge import KNOWLEDGE_PRIMER
        parts.append("\n" + KNOWLEDGE_PRIMER)
    except Exception as e:
        logger.warning("Could not load knowledge primer: %s", e)

    # 6. Bioinformatics code-gen hints (CRITICAL for BixBench performance)
    try:
        from ct.tools.code import BIOINFORMATICS_CODE_GEN_PROMPT, AGENTIC_CODE_ADDENDUM
        # Strip the template placeholders and include the raw hints
        hints = BIOINFORMATICS_CODE_GEN_PROMPT
        # Remove the {namespace_description} and {data_files_description} placeholders
        hints = hints.replace("{namespace_description}", "(see run_python tool description)")
        hints = hints.replace("{data_files_description}", "(see data context below)")
        parts.append("\n## Bioinformatics Code Generation Guide\n")
        parts.append(
            "When using ``run_python`` for bioinformatics analysis, follow these "
            "patterns and guidelines:\n"
        )
        parts.append(hints)
        parts.append(AGENTIC_CODE_ADDENDUM)
    except Exception as e:
        logger.warning("Could not load code-gen hints: %s", e)

    # 7. Synthesis rules
    try:
        from ct.agent.knowledge import SYNTHESIZER_PRIMER
        parts.append("\n## Synthesis Guidelines\n")
        parts.append(SYNTHESIZER_PRIMER)
    except Exception as e:
        logger.warning("Could not load synthesizer primer: %s", e)

    # 8. Synthesis instructions
    parts.append(_SYNTHESIS_INSTRUCTIONS)

    # 8b. Structure input guidance
    parts.append(_STRUCTURE_INPUT_GUIDANCE)

    # 9. Dynamic data context
    if data_context:
        parts.append("\n## Data Context\n")
        parts.append(data_context)

    # 10. Session history (for multi-turn interactive mode)
    if history:
        parts.append("\n## Prior Conversation\n")
        parts.append(history)

    prompt = "\n".join(parts)
    logger.info(
        "Built system prompt: %d chars, %d sections",
        len(prompt),
        len(parts),
    )
    return prompt
