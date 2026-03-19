#!/usr/bin/env python3
"""
related_research_finder — CLI entry point

Usage examples
--------------

  # By S2 paper ID
  python main.py 204e3073870fae3d05bcbc2f6a8e263d9b72e776

  # By arXiv ID
  python main.py "arXiv:1706.03762"

  # By DOI
  python main.py "DOI:10.18653/v1/N18-1202"

  # By title (fuzzy search fallback)
  python main.py "Attention is all you need"

  # Include papers that cite the target; save results to JSON
  python main.py "arXiv:1706.03762" --include-citations --output results.json

  # Provide your S2 API key to lift rate limits (optional)
  python main.py "arXiv:1706.03762" --api-key YOUR_KEY_HERE
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from finder import FinderResult, RelatedPaper, RelatedPaperFinder
from s2_client import S2Client


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _paper_to_dict(p: RelatedPaper) -> dict:
    return {
        "paperId": p.paper_id,
        "title": p.title,
        "year": p.year,
        "authors": p.authors,
        "venue": p.venue,
        "citationCount": p.citation_count,
        "abstract": p.abstract,
        "url": p.url,
        "pdfUrl": p.pdf_url,
        "externalIds": p.external_ids,
        "sources": p.sources,
    }


def _result_to_dict(result: FinderResult) -> dict:
    return {
        "targetPaper": _paper_to_dict(result.target_paper),
        "relatedPapers": [_paper_to_dict(p) for p in result.related],
        "stats": {
            "total": len(result.related),
            "fromReferences": len(result.by_source("reference")),
            "fromRecommendations": len(result.by_source("recommendation")),
            "fromCitations": len(result.by_source("citation")),
            "foundByMultipleSources": sum(
                1 for p in result.related if len(p.sources) > 1
            ),
        },
    }


def _print_summary(result: FinderResult, top_n: int = 20) -> None:
    t = result.target_paper
    print("\n" + "=" * 72)
    print("TARGET PAPER")
    print("=" * 72)
    print(f"  Title  : {t.title}")
    print(f"  Year   : {t.year}")
    print(f"  Authors: {t.short_authors()}")
    print(f"  URL    : {t.url}")
    if t.pdf_url:
        print(f"  PDF    : {t.pdf_url}")

    stats = _result_to_dict(result)["stats"]
    print(f"\n  Related papers found : {stats['total']}")
    print(f"    From references    : {stats['fromReferences']}")
    print(f"    From recommendations : {stats['fromRecommendations']}")
    if stats["fromCitations"]:
        print(f"    From citations     : {stats['fromCitations']}")
    print(f"    Multi-source       : {stats['foundByMultipleSources']}")

    print(f"\n{'─' * 72}")
    print(f"TOP {top_n} RELATED PAPERS  (sorted by citation count)")
    print(f"{'─' * 72}")
    for i, p in enumerate(result.top_cited(top_n), 1):
        src_tag = "+".join(p.sources)
        arxiv = p.arxiv_id()
        id_str = f"arXiv:{arxiv}" if arxiv else p.paper_id[:12] + "…"
        print(
            f"{i:>3}. [{src_tag:>22}]  {p.citation_count:>6} cites | "
            f"{p.year or '????'} | {id_str}"
        )
        print(f"       {p.title[:70]}")
        print(f"       {p.short_authors()}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Find research papers related to a target S2ORC paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "paper",
        help=(
            "Target paper identifier: S2 paper ID, arXiv:ID, DOI:xxx, "
            "CorpusID:NNN, or a free-text title."
        ),
    )
    p.add_argument(
        "--api-key",
        metavar="KEY",
        default=None,
        help="Semantic Scholar API key (optional – raises rate limit).",
    )
    p.add_argument(
        "--max-references",
        type=int,
        default=200,
        metavar="N",
        help="Maximum number of references to retrieve (default: 200).",
    )
    p.add_argument(
        "--max-recommendations",
        type=int,
        default=100,
        metavar="N",
        help="Maximum number of S2 recommended papers (default: 100).",
    )
    p.add_argument(
        "--include-citations",
        action="store_true",
        default=False,
        help="Also include papers that cite the target paper.",
    )
    p.add_argument(
        "--max-citations",
        type=int,
        default=100,
        metavar="N",
        help="Max incoming citations to retrieve if --include-citations is set.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of top papers to show in the summary (default: 20).",
    )
    p.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        default=None,
        help="Save full results as JSON to FILE.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Always show INFO from finder/client when verbose
    if args.verbose:
        logging.getLogger("finder").setLevel(logging.DEBUG)
        logging.getLogger("s2_client").setLevel(logging.DEBUG)

    client = S2Client(api_key=args.api_key)
    finder = RelatedPaperFinder(
        client=client,
        max_references=args.max_references,
        max_recommendations=args.max_recommendations,
        include_citations=args.include_citations,
        max_citations=args.max_citations,
    )

    try:
        result = finder.find(args.paper)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_summary(result, top_n=args.top)

    if args.output:
        out = _result_to_dict(result)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
