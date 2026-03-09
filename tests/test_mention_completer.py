"""Tests for MentionCompleter, MergedCompleter, and mention parsing functions."""

import pytest
from prompt_toolkit.document import Document

from ct.ui.terminal import (
    MentionCompleter,
    MergedCompleter,
    SlashCompleter,
    DATASET_CANDIDATES,
    extract_mentions,
    build_mention_context,
)


# ---------------------------------------------------------------------------
# Test candidates
# ---------------------------------------------------------------------------

SAMPLE_CANDIDATES = [
    ("target.coessentiality", "target", "Find co-essential gene partners", "tool"),
    ("target.neosubstrate_score", "target", "Score neosubstrate potential", "tool"),
    ("target.degron_motif", "target", "Scan for degron motifs", "tool"),
    ("expression.pathway_enrichment", "expression", "Gene set enrichment analysis", "tool"),
    ("expression.l1000_signature", "expression", "L1000 expression signature", "tool"),
    ("viability.dose_response", "viability", "Dose-response curve analysis", "tool"),
    ("depmap", "dataset", "DepMap CRISPR/model data", "dataset"),
    ("prism", "dataset", "PRISM drug sensitivity", "dataset"),
    ("l1000", "dataset", "L1000 gene expression signatures", "dataset"),
]


def _completions(doc_text: str, candidates=None):
    """Helper: get completions as a list of (text, style) tuples."""
    completer = MentionCompleter(candidates or SAMPLE_CANDIDATES)
    doc = Document(doc_text, len(doc_text))
    results = list(completer.get_completions(doc, None))
    return [(c.text, getattr(c, "style", "")) for c in results]


# ---------------------------------------------------------------------------
# MentionCompleter
# ---------------------------------------------------------------------------

class TestMentionCompleter:
    def test_trigger_on_at(self):
        results = _completions("@")
        assert len(results) > 0

    def test_partial_tool_name(self):
        results = _completions("@targ")
        names = [r[0] for r in results]
        assert "@target.coessentiality" in names
        assert "@target.neosubstrate_score" in names
        assert "@target.degron_motif" in names

    def test_fuzzy_match(self):
        results = _completions("@coessen")
        names = [r[0] for r in results]
        assert "@target.coessentiality" in names

    def test_category_prefix(self):
        results = _completions("@target")
        names = [r[0] for r in results]
        # All target tools should match
        assert "@target.coessentiality" in names
        assert "@target.neosubstrate_score" in names
        assert "@target.degron_motif" in names
        # Non-target tools should not match (unless "target" is in their desc)
        assert "@expression.pathway_enrichment" not in names

    def test_dataset_match(self):
        results = _completions("@dep")
        names = [r[0] for r in results]
        assert "@depmap" in names

    def test_no_matches(self):
        results = _completions("@zzzznonexistent")
        assert len(results) == 0

    def test_tool_style(self):
        results = _completions("@coessen")
        for text, style in results:
            if "coessentiality" in text:
                assert style == "class:mention-tool"

    def test_dataset_style(self):
        results = _completions("@dep")
        for text, style in results:
            if "depmap" in text:
                assert style == "class:mention-dataset"

    def test_no_trigger_without_at(self):
        results = _completions("analyze CRBN")
        assert len(results) == 0

    def test_at_after_text(self):
        results = _completions("analyze CRBN @targ")
        names = [r[0] for r in results]
        assert "@target.coessentiality" in names

    def test_description_match(self):
        results = _completions("@enrichment")
        names = [r[0] for r in results]
        assert "@expression.pathway_enrichment" in names


# ---------------------------------------------------------------------------
# MergedCompleter
# ---------------------------------------------------------------------------

class TestMergedCompleter:
    def test_slash_commands(self):
        merged = MergedCompleter(
            SlashCompleter(),
            MentionCompleter(SAMPLE_CANDIDATES),
        )
        doc = Document("/tool", len("/tool"))
        results = list(merged.get_completions(doc, None))
        names = [c.text for c in results]
        assert "/tools" in names

    def test_at_mentions(self):
        merged = MergedCompleter(
            SlashCompleter(),
            MentionCompleter(SAMPLE_CANDIDATES),
        )
        doc = Document("@targ", len("@targ"))
        results = list(merged.get_completions(doc, None))
        names = [c.text for c in results]
        assert "@target.coessentiality" in names

    def test_no_trigger(self):
        merged = MergedCompleter(
            SlashCompleter(),
            MentionCompleter(SAMPLE_CANDIDATES),
        )
        doc = Document("analyze CRBN", len("analyze CRBN"))
        results = list(merged.get_completions(doc, None))
        assert len(results) == 0


# ---------------------------------------------------------------------------
# extract_mentions
# ---------------------------------------------------------------------------

class TestExtractMentions:
    def test_single_tool(self):
        query, tools, datasets, _ = extract_mentions("analyze CRBN @target.coessentiality")
        assert query == "analyze CRBN"
        assert tools == ["target.coessentiality"]
        assert datasets == []

    def test_single_dataset(self):
        query, tools, datasets, _ = extract_mentions("check sensitivity @depmap")
        assert query == "check sensitivity"
        assert datasets == ["depmap"]
        assert tools == []

    def test_multiple_mixed(self):
        query, tools, datasets, _ = extract_mentions(
            "analyze @target.coessentiality @expression.pathway_enrichment @depmap"
        )
        assert query == "analyze"
        assert "target.coessentiality" in tools
        assert "expression.pathway_enrichment" in tools
        assert "depmap" in datasets

    def test_no_mentions(self):
        query, tools, datasets, _ = extract_mentions("analyze CRBN")
        assert query == "analyze CRBN"
        assert tools == []
        assert datasets == []

    def test_at_end_of_string(self):
        query, tools, datasets, _ = extract_mentions("test @")
        assert "test" in query
        assert tools == []

    def test_double_at(self):
        query, tools, datasets, _ = extract_mentions("test @@depmap")
        # Should handle gracefully — one @depmap extracted
        assert "depmap" in datasets or tools == []

    def test_unknown_dataset_ignored(self):
        query, tools, datasets, _ = extract_mentions("test @randomword")
        assert datasets == []


# ---------------------------------------------------------------------------
# build_mention_context
# ---------------------------------------------------------------------------

class TestBuildMentionContext:
    def test_tool_context(self):
        ctx = build_mention_context(["target.coessentiality"], [])
        assert "MUST include" in ctx
        assert "target.coessentiality" in ctx

    def test_dataset_context(self):
        ctx = build_mention_context([], ["depmap"])
        assert "depmap" in ctx
        assert "DepMap" in ctx

    def test_combined(self):
        ctx = build_mention_context(
            ["target.coessentiality"],
            ["depmap", "prism"],
        )
        assert "MUST include" in ctx
        assert "depmap" in ctx
        assert "prism" in ctx

    def test_empty(self):
        ctx = build_mention_context([], [])
        assert ctx == ""
