"""
Semantic Scholar API client.

Wraps the S2AG (Semantic Scholar Academic Graph) REST API and the
Recommendations API.  Both are free and need no API key, though providing
one raises the rate-limit significantly.

Docs: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_BASE = "https://api.semanticscholar.org/recommendations/v1"

# Fields we want for every paper object returned by the graph API
PAPER_FIELDS = ",".join(
    [
        "paperId",
        "externalIds",
        "url",
        "title",
        "abstract",
        "year",
        "authors",
        "venue",
        "publicationTypes",
        "citationCount",
        "referenceCount",
        "openAccessPdf",
    ]
)

# Fields to request when paginating references / citations
EDGE_FIELDS = (
    f"paperId,title,abstract,year,authors,venue,citationCount,externalIds,openAccessPdf"
)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _build_session(api_key: str | None = None) -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    session.headers.update(headers)
    return session


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class S2Client:
    """Thin, rate-aware wrapper around the Semantic Scholar public APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        requests_per_second: float = 0.5,
    ) -> None:
        self._session = _build_session(api_key)
        self._min_interval = (
            1.0 / requests_per_second if requests_per_second > 0 else 0.0
        )
        self._last_call_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.monotonic()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        self._throttle()
        logger.debug("GET %s params=%s", url, params)
        response = self._session.get(url, params=params, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Graph API — single paper
    # ------------------------------------------------------------------

    def get_paper(self, paper_id: str) -> dict | None:
        """
        Fetch a single paper.

        paper_id may be any of:
          - bare S2 paperId  (40-char hex)
          - "arXiv:<id>"     e.g. arXiv:2301.00001
          - "DOI:<doi>"
          - "CorpusID:<int>"
          - "PMID:<int>"
        """
        url = f"{GRAPH_BASE}/paper/{paper_id}"
        data = self._get(url, params={"fields": PAPER_FIELDS})
        return data

    def search_papers(self, query: str, limit: int = 5) -> list[dict]:
        """Full-text search; return up to *limit* result papers."""
        url = f"{GRAPH_BASE}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": PAPER_FIELDS,
        }
        data = self._get(url, params=params)
        if data is None:
            return []
        return data.get("data", [])

    # ------------------------------------------------------------------
    # Graph API — references & citations (paginated)
    # ------------------------------------------------------------------

    def get_references(self, paper_id: str, limit: int = 500) -> list[dict]:
        """
        Papers that *paper_id* cites (its reference list).
        Returns a flat list of paper dicts.
        """
        return self._paginate(
            f"{GRAPH_BASE}/paper/{paper_id}/references",
            fields=EDGE_FIELDS,
            limit=limit,
            nested_key="citedPaper",
        )

    def get_citations(self, paper_id: str, limit: int = 500) -> list[dict]:
        """
        Papers that cite *paper_id*.
        Returns a flat list of paper dicts.
        """
        return self._paginate(
            f"{GRAPH_BASE}/paper/{paper_id}/citations",
            fields=EDGE_FIELDS,
            limit=limit,
            nested_key="citingPaper",
        )

    def _paginate(
        self,
        url: str,
        fields: str,
        limit: int,
        nested_key: str,
    ) -> list[dict]:
        results: list[dict] = []
        offset = 0
        page_size = min(limit, 1000)

        while len(results) < limit:
            params = {
                "fields": fields,
                "offset": offset,
                "limit": page_size,
            }
            data = self._get(url, params=params)
            if not data:
                break
            page = data.get("data", [])
            for edge in page:
                paper = edge.get(nested_key)
                if paper and paper.get("paperId"):
                    results.append(paper)
            if len(page) < page_size or data.get("next") is None:
                break
            offset = data["next"]

        return results[:limit]

    # ------------------------------------------------------------------
    # Recommendations API
    # ------------------------------------------------------------------

    def get_recommendations(self, paper_id: str, limit: int = 100) -> list[dict]:
        """
        Papers recommended by Semantic Scholar for *paper_id*.

        Uses POST /recommendations/v1/papers with a single positive paper ID.
        This endpoint reliably returns results where the GET forpaper endpoint
        sometimes returns empty.
        """
        self._throttle()
        url = f"{RECOMMENDATIONS_BASE}/papers"
        params = {"fields": PAPER_FIELDS, "limit": min(limit, 500)}
        payload = {
            "positivePaperIds": [paper_id],
            "negativePaperIds": [],
        }
        logger.debug("POST %s params=%s", url, params)
        response = self._session.post(url, params=params, json=payload, timeout=30)
        self._last_call_time = time.monotonic()
        if response.status_code == 404:
            return []
        response.raise_for_status()
        data = response.json()
        return data.get("recommendedPapers", [])
