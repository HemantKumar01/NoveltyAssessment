"""
Core logic: given a target paper, collect related papers from two sources:

  1. References  — papers that the target paper cites
  2. Recommendations — papers returned by the S2 Recommendations API

Results are deduplicated and enriched with a provenance tag so callers know
how each paper was surfaced.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Literal

from s2_client import S2Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

Source = Literal["reference", "recommendation", "citation"]


@dataclasses.dataclass
class RelatedPaper:
    paper_id: str
    title: str
    year: int | None
    authors: list[str]
    venue: str
    citation_count: int
    abstract: str
    url: str
    pdf_url: str | None
    external_ids: dict
    sources: list[Source]  # may contain multiple if found by both methods

    # Convenience -----------------------------------------------------------

    def arxiv_id(self) -> str | None:
        return self.external_ids.get("ArXiv")

    def doi(self) -> str | None:
        return self.external_ids.get("DOI")

    def short_authors(self, max_authors: int = 3) -> str:
        if not self.authors:
            return "Unknown"
        names = self.authors[:max_authors]
        suffix = " et al." if len(self.authors) > max_authors else ""
        return ", ".join(names) + suffix


@dataclasses.dataclass
class FinderResult:
    target_paper: RelatedPaper
    related: list[RelatedPaper]

    # Convenience filters ---------------------------------------------------

    def by_source(self, source: Source) -> list[RelatedPaper]:
        return [p for p in self.related if source in p.sources]

    def top_cited(self, n: int = 20) -> list[RelatedPaper]:
        return sorted(self.related, key=lambda p: p.citation_count, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_authors(raw_authors: list[dict]) -> list[str]:
    return [a.get("name", "") for a in (raw_authors or []) if a.get("name")]


def _to_related_paper(raw: dict, source: Source) -> RelatedPaper | None:
    pid = raw.get("paperId")
    title = raw.get("title") or ""
    if not pid or not title.strip():
        return None

    authors = _extract_authors(raw.get("authors") or [])
    returning = RelatedPaper(
        paper_id=pid,
        title=title.strip(),
        year=raw.get("year"),
        authors=authors,
        venue=raw.get("venue") or "",
        citation_count=raw.get("citationCount") or 0,
        abstract=(raw.get("abstract") or "").strip(),
        url=raw.get("url") or f"https://www.semanticscholar.org/paper/{pid}",
        pdf_url=(raw.get("openAccessPdf") or {}).get("url"),
        external_ids=raw.get("externalIds") or {},
        sources=[source],
    )
    return returning


def _merge_into(
    registry: dict[str, RelatedPaper],
    papers: list[dict],
    source: Source,
) -> None:
    """Insert or update papers in registry, merging sources list."""
    for raw in papers:
        rp = _to_related_paper(raw, source)
        if rp is None:
            continue
        if rp.paper_id in registry:
            if source not in registry[rp.paper_id].sources:
                registry[rp.paper_id].sources.append(source)
        else:
            registry[rp.paper_id] = rp


# ---------------------------------------------------------------------------
# Main finder
# ---------------------------------------------------------------------------


class RelatedPaperFinder:
    """
    Finds papers related to a given target paper.

    Parameters
    ----------
    client:
        An S2Client instance (handles rate limiting, retries).
    max_references:
        Maximum number of references (cited papers) to retrieve.
    max_recommendations:
        Maximum number of API-recommended papers to retrieve.
    include_citations:
        If True, also include papers that *cite* the target paper (incoming
        citations).  Defaults to False to keep results focused on content
        similarity rather than popularity.
    max_citations:
        Maximum number of citing papers to retrieve (only used when
        include_citations=True).
    """

    def __init__(
        self,
        client: S2Client,
        max_references: int = 200,
        max_recommendations: int = 100,
        include_citations: bool = False,
        max_citations: int = 100,
    ) -> None:
        self._client = client
        self._max_references = max_references
        self._max_recommendations = max_recommendations
        self._include_citations = include_citations
        self._max_citations = max_citations

    # ------------------------------------------------------------------
    # Resolve paper identifier → canonical ID
    # ------------------------------------------------------------------

    def resolve(self, identifier: str) -> dict | None:
        """
        Accept:
          - A bare Semantic Scholar paperId
          - arXiv:XXXX.XXXXX
          - DOI:10.xxx/yyy
          - CorpusID:NNN
          - Free-text title (fallback: use title search)
        """
        # Prefixed IDs are passed straight through to the API
        prefixed = ("arXiv:", "DOI:", "PMID:", "CorpusID:")
        if any(identifier.startswith(p) for p in prefixed):
            return self._client.get_paper(identifier)

        # 40-char hex → bare S2 paper ID
        if len(identifier) == 40 and all(
            c in "0123456789abcdefABCDEF" for c in identifier
        ):
            return self._client.get_paper(identifier)

        # Fallback: title search
        logger.info("Identifier looks like a title – searching for: %s", identifier)
        results = self._client.search_papers(identifier, limit=3)
        if not results:
            return None
        best = results[0]
        logger.info(
            "Resolved to: %s (paperId=%s)", best.get("title"), best.get("paperId")
        )
        return best

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def find(self, identifier: str) -> FinderResult:
        """
        Given an identifier for the target paper, return a FinderResult with
        all related papers collected from references and recommendations.
        """
        # 1. Resolve target
        logger.info("Resolving target paper: %s", identifier)
        target_raw = self.resolve(identifier)
        if target_raw is None:
            raise ValueError(f"Could not resolve paper: {identifier!r}")

        pid = target_raw["paperId"]
        logger.info("Target paper ID: %s | Title: %s", pid, target_raw.get("title"))

        target = _to_related_paper(target_raw, "reference")
        assert target is not None
        target.sources = []  # target is not "related to itself"

        registry: dict[str, RelatedPaper] = {}

        # 2. References (papers cited by the target)
        logger.info("Fetching references (cited papers)…")
        refs = self._client.get_references(pid, limit=self._max_references)
        logger.info("  → %d references", len(refs))
        _merge_into(registry, refs, "reference")

        # 3. Recommendations from S2 API
        logger.info("Fetching S2 recommendations…")
        recs = self._client.get_recommendations(pid, limit=self._max_recommendations)
        logger.info("  → %d recommendations", len(recs))
        _merge_into(registry, recs, "recommendation")

        # 4. (Optional) Incoming citations
        if self._include_citations:
            logger.info("Fetching citing papers…")
            cites = self._client.get_citations(pid, limit=self._max_citations)
            logger.info("  → %d citing papers", len(cites))
            _merge_into(registry, cites, "citation")

        # Remove target itself from results (can appear in recommendations)
        registry.pop(pid, None)

        related = sorted(
            registry.values(),
            key=lambda p: p.citation_count,
            reverse=True,
        )
        logger.info("Total unique related papers: %d", len(related))

        return FinderResult(target_paper=target, related=related)
