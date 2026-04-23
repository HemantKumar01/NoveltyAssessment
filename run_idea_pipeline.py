#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Allow importing existing modules that are organized in folders.
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "extract_motivation"))
sys.path.append(str(ROOT / "inspirationPapersFinder"))
sys.path.append(str(ROOT / "NoveltyAssessment"))

from extractor import extract_arise_features, extract_text_from_pdf
from google import genai
from google.genai import types
from novelty_scorer import compute_novelty_score
from s2_client import S2Client


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end pipeline: recent paper selection, topic/motivation "
            "extraction, idea generation, novelty scoring, and ranking."
        )
    )
    parser.add_argument(
        "--query",
        default="large language models for scientific discovery",
        help="Semantic Scholar query used to pick a recent paper.",
    )
    parser.add_argument(
        "--paper-id",
        default=None,
        help=(
            "Optional explicit paper identifier (paperId, arXiv:..., DOI:...). "
            "If provided, query-based selection is skipped."
        ),
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2024,
        help="Minimum publication year for automatic paper selection.",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=25,
        help="How many papers to inspect when selecting a recent paper.",
    )
    parser.add_argument(
        "--ideas",
        type=int,
        default=10,
        help="Number of ideas to generate.",
    )
    parser.add_argument(
        "--papers-per-idea",
        type=int,
        default=15,
        help="How many Semantic Scholar papers to retrieve per keyword for scoring.",
    )
    parser.add_argument(
        "--output",
        default="pipeline_results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--fallback-related-json",
        default="inspirationPapersFinder/results.json",
        help=(
            "Fallback JSON containing related papers for novelty scoring "
            "when Semantic Scholar API calls are rate-limited."
        ),
    )
    parser.add_argument(
        "--pdf-path",
        default=None,
        help="Optional local PDF path to use instead of downloading open access PDF.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def select_recent_paper(
    client: S2Client, query: str, min_year: int, search_limit: int
) -> dict:
    papers = client.search_papers(query=query, limit=search_limit)
    if not papers:
        raise ValueError("No papers found from Semantic Scholar search")

    candidates = [p for p in papers if (p.get("year") or 0) >= min_year]
    if not candidates:
        raise ValueError(f"No papers found with year >= {min_year}")

    # Pick the most cited recent paper for stable quality.
    selected = max(candidates, key=lambda p: p.get("citationCount", 0))
    return selected


def select_recent_arxiv_paper(query: str) -> dict:
    encoded_query = requests.utils.quote(query)
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded_query}&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", namespace)
    if not entries:
        raise ValueError("No papers found from arXiv API")

    entry = entries[0]
    title = (
        entry.findtext("atom:title", default="", namespaces=namespace) or ""
    ).strip()
    abstract = (
        entry.findtext("atom:summary", default="", namespaces=namespace) or ""
    ).strip()
    published = entry.findtext("atom:published", default="", namespaces=namespace)
    year = int(published[:4]) if published and len(published) >= 4 else None

    pdf_url = None
    for link in entry.findall("atom:link", namespace):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href")
            break

    paper_id = (
        entry.findtext("atom:id", default="", namespaces=namespace) or ""
    ).strip()

    return {
        "paperId": paper_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "venue": "arXiv",
        "citationCount": 0,
        "url": paper_id,
        "openAccessPdf": {"url": pdf_url} if pdf_url else {},
    }


def fetch_full_paper(client: S2Client, paper_id: str) -> dict:
    paper = client.get_paper(paper_id)
    if not paper:
        raise ValueError(f"Could not fetch paper details for identifier: {paper_id}")
    return paper


def download_pdf(url: str, target_path: Path) -> Path | None:
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        return target_path
    except Exception as exc:
        logger.warning("PDF download failed (%s): %s", url, exc)
        return None


def build_input_text_for_extraction(paper: dict, local_pdf: Path | None) -> str:
    if local_pdf and local_pdf.exists():
        extracted = extract_text_from_pdf(str(local_pdf))
        if extracted and extracted.strip():
            return extracted

    # Fallback to metadata text if PDF is unavailable.
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    venue = (paper.get("venue") or "").strip()
    year = paper.get("year")
    return (
        f"Title: {title}\n"
        f"Year: {year}\n"
        f"Venue: {venue}\n"
        f"Abstract: {abstract}\n"
    )


def generate_ideas(topic: str, motivation: str, num_ideas: int) -> list[dict[str, str]]:
    client = genai.Client()
    system_prompt = (
        "You generate concrete, novel research ideas in JSON only. "
        "Each idea must include a concise title and a detailed description "
        "with problem, method direction, and expected evaluation."
    )
    user_prompt = (
        "Given the research context below, generate exactly "
        f"{num_ideas} research ideas.\n\n"
        f"Topic: {topic}\n\n"
        f"Motivation: {motivation}\n\n"
        "Return JSON with this exact structure:\n"
        "{\n"
        '  "ideas": [\n'
        "    {\n"
        '      "title": "...",\n'
        '      "description": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.7,
        ),
    )
    parsed = json.loads(response.text)
    ideas = parsed.get("ideas", [])
    if len(ideas) < num_ideas:
        raise ValueError(f"Expected {num_ideas} ideas, got {len(ideas)}")

    return ideas[:num_ideas]


def load_fallback_related_papers(path: Path) -> list[dict]:
    if not path.exists():
        logger.warning("Fallback related papers file not found: %s", path)
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    papers = payload.get("relatedPapers", [])
    if not isinstance(papers, list):
        return []

    return papers


def score_ideas(
    ideas: list[dict[str, str]],
    s2_client: S2Client,
    papers_per_keyword: int,
    fallback_papers: list[dict],
) -> list[dict[str, Any]]:
    scored = []

    for idx, idea in enumerate(ideas, start=1):
        title = (idea.get("title") or "").strip()
        description = (idea.get("description") or "").strip()
        idea_text = f"{title}. {description}".strip()

        logger.info("Scoring idea %d/%d", idx, len(ideas))

        # Prefer API retrieval; fall back to local cached papers if available.
        papers_input = None
        client_input = s2_client
        if fallback_papers:
            papers_input = fallback_papers
            client_input = None

        result = compute_novelty_score(
            generated_idea=idea_text,
            papers=papers_input,
            s2_client=client_input,
            keywords=None,
            papers_per_keyword=papers_per_keyword,
            embedding_model="all-MiniLM-L6-v2",
        )

        result["title"] = title
        result["description"] = description
        result["idea_text"] = idea_text
        result["papers_per_keyword"] = papers_per_keyword
        scored.append(result)

    scored.sort(key=lambda x: x.get("novelty_score", -1.0), reverse=True)
    return scored


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    load_dotenv()

    s2_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    s2_client = S2Client(api_key=s2_api_key)
    fallback_papers = load_fallback_related_papers(Path(args.fallback_related_json))

    if args.paper_id:
        selected = fetch_full_paper(s2_client, args.paper_id)
    else:
        try:
            selected = select_recent_paper(
                s2_client,
                query=args.query,
                min_year=args.min_year,
                search_limit=args.search_limit,
            )
            selected = fetch_full_paper(s2_client, selected["paperId"])
        except Exception as exc:
            logger.warning(
                "Semantic Scholar selection failed (%s). Falling back to arXiv.",
                exc,
            )
            selected = select_recent_arxiv_paper(args.query)

    logger.info(
        "Selected paper: %s (%s)",
        selected.get("title"),
        selected.get("year"),
    )

    pdf_path: Path | None = None
    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
    else:
        pdf_url = (selected.get("openAccessPdf") or {}).get("url")
        if pdf_url:
            pdf_path = download_pdf(pdf_url, ROOT / "selected_paper.pdf")

    extraction_input = build_input_text_for_extraction(selected, pdf_path)
    extracted = extract_arise_features(extraction_input)
    if not extracted:
        raise RuntimeError("Failed to extract topic and motivation")

    topic = (extracted.get("topic") or "").strip()
    motivation = (extracted.get("motivation") or "").strip()

    if not topic or not motivation:
        raise RuntimeError("Extraction output missing topic or motivation")

    logger.info("Generating %d ideas", args.ideas)
    ideas = generate_ideas(topic=topic, motivation=motivation, num_ideas=args.ideas)

    scored_ideas = score_ideas(
        ideas=ideas,
        s2_client=s2_client,
        papers_per_keyword=args.papers_per_idea,
        fallback_papers=fallback_papers,
    )

    output = {
        "selected_paper": {
            "paperId": selected.get("paperId"),
            "title": selected.get("title"),
            "year": selected.get("year"),
            "url": selected.get("url"),
            "pdfUrl": (selected.get("openAccessPdf") or {}).get("url"),
        },
        "extraction": {
            "topic": topic,
            "motivation": motivation,
        },
        "ideas_sorted_by_novelty": scored_ideas,
    }

    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Pipeline output written to %s", out_path)

    print("\nTop ideas by novelty score:")
    for rank, item in enumerate(scored_ideas, start=1):
        print(
            f"{rank:>2}. {item.get('title', 'Untitled')} | score={item.get('novelty_score', 0.0):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
