"""
Data Collection Module for Novel vs Non-Novel Paper Classification

Pipeline:
1. Fetch papers from Semantic Scholar
2. Extract the core "idea" from each paper using Vertex AI (Gemini)
3. Collect related papers using similarity + citation retrieval
4. Label: Novel (1) or Not-Novel (0) based on retrieval strategy
5. Output a JSONL dataset
"""

import os
import time
import json
import logging
import argparse
import requests
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from dotenv import load_dotenv

load_dotenv()


# ─── Config ────────────────────────────────────────────────────────────────────
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_REC_BASE = "https://api.semanticscholar.org/recommendations/v1"
S2_API_KEY = os.environ.get("S2_API_KEY", "")

# Vertex AI settings — override via env vars or CLI args
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "")  # GCP project ID
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-2.5-pro")

# Set GOOGLE_APPLICATION_CREDENTIALS env var to your service-account JSON path,
# OR pass --vertex-api-key (path to service-account JSON) on the CLI.

PAPER_FIELDS = (
    "paperId,title,abstract,year,authors,references,citations,publicationVenue"
)
REF_FIELDS = "paperId,title,abstract,year"

TOP_VENUES = {
    "NeurIPS",
    "ICML",
    "ICLR",
    "ACL",
    "EMNLP",
    "NAACL",
    "CVPR",
    "ICCV",
    "ECCV",
    "AAAI",
    "IJCAI",
    "KDD",
    "WWW",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Vertex AI client (lazy-initialised once) ───────────────────────────────────
_vertex_model: Optional[GenerativeModel] = None


def get_vertex_model() -> GenerativeModel:
    global _vertex_model
    if _vertex_model is None:
        if not VERTEX_PROJECT:
            raise RuntimeError(
                "VERTEX_PROJECT is not set. "
                "Pass --vertex-project or set the VERTEX_PROJECT env var."
            )
        vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        _vertex_model = GenerativeModel(VERTEX_MODEL)
        log.info(
            "Vertex AI initialised: project=%s location=%s model=%s",
            VERTEX_PROJECT,
            VERTEX_LOCATION,
            VERTEX_MODEL,
        )
    return _vertex_model


def vertex_generate(
    prompt: str, max_tokens: int = 512, temperature: float = 0.2
) -> str:
    """Call Vertex AI Gemini and return the text response."""
    model = get_vertex_model()
    config = GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    response = model.generate_content(prompt, generation_config=config)
    return response.text.strip()


# ─── Semantic Scholar helpers ───────────────────────────────────────────────────


def s2_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if S2_API_KEY:
        h["x-api-key"] = S2_API_KEY
    return h


def s2_get(path: str, params: dict = None, retries: int = 3) -> Optional[dict]:
    url = S2_BASE + path
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=s2_headers(), timeout=20)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning("Rate-limited; waiting %ds", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning(
                "S2 request failed (%s); attempt %d/%d", e, attempt + 1, retries
            )
            time.sleep(3)
    return None


def s2_search(query: str, limit: int = 10) -> list[dict]:
    data = s2_get(
        "/paper/search", {"query": query, "limit": limit, "fields": PAPER_FIELDS}
    )
    return data.get("data", []) if data else []


def s2_paper(paper_id: str) -> Optional[dict]:
    return s2_get(f"/paper/{paper_id}", {"fields": PAPER_FIELDS})


def s2_recommendations(paper_id: str, limit: int = 10) -> list[dict]:
    """Similarity-based retrieval via S2 recommendations endpoint."""
    url = f"{S2_REC_BASE}/papers/forpaper/{paper_id}"
    for attempt in range(3):
        try:
            r = requests.get(
                url,
                params={"limit": limit, "fields": REF_FIELDS},
                headers=s2_headers(),
                timeout=20,
            )
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 404:
                log.warning("No recommendations found for paper %s", paper_id)
                return []
            r.raise_for_status()
            return r.json().get("recommendedPapers", [])
        except requests.RequestException as e:
            log.warning(
                "Recommendations request failed (%s); attempt %d/3", e, attempt + 1
            )
            time.sleep(3)
    return []


def fetch_bulk_papers(paper_ids: list[str]) -> list[dict]:
    """Batch-fetch paper details."""
    url = S2_BASE + "/paper/batch"
    payload = {"ids": paper_ids}
    params = {"fields": REF_FIELDS}
    try:
        r = requests.post(
            url, json=payload, params=params, headers=s2_headers(), timeout=30
        )
        r.raise_for_status()
        return [p for p in r.json() if p]
    except Exception as e:
        log.warning("Batch fetch failed: %s", e)
        return []


def get_citation_neighbours(
    paper: dict, target_year: int, limit: int = 10
) -> list[dict]:
    """
    Citation-based retrieval:
      - References of P (papers cited by P, published before P)
    """
    neighbours = []
    for ref in (paper.get("references") or [])[: limit * 2]:
        ref_id = ref.get("paperId")
        if not ref_id:
            continue
        detail = s2_paper(ref_id)
        if detail and detail.get("year") and detail["year"] < target_year:
            neighbours.append(detail)
        if len(neighbours) >= limit:
            break
    return neighbours[:limit]


# ─── Idea extraction via Vertex AI ─────────────────────────────────────────────


def extract_idea(title: str, abstract: str) -> str:
    """Use Vertex AI Gemini to extract the core idea/contribution from a paper."""
    prompt = (
        f"Title: {title}\n\nAbstract: {abstract}\n\n"
        "In 2-4 sentences, summarize the single core idea or novel contribution of this paper. "
        "Be concise, specific, and technical. Do not add any preamble or explanation."
    )
    return vertex_generate(prompt, max_tokens=300, temperature=0.1)


# ─── Related-paper filtering via Vertex AI ──────────────────────────────────────


def filter_related_by_idea(
    idea: str, candidates: list[dict], top_k: int = 10
) -> list[dict]:
    """
    Ask Vertex AI Gemini to pick the top_k most relevant papers to the given idea.
    Returns a filtered + ranked list.
    """
    if not candidates:
        return []

    numbered = "\n".join(
        f"{i+1}. [{p.get('paperId','')}] {p.get('title','')}: {(p.get('abstract') or '')[:300]}"
        for i, p in enumerate(candidates)
    )
    prompt = (
        f"Core idea of target paper:\n{idea}\n\n"
        f"Candidate related papers:\n{numbered}\n\n"
        f"Select the {top_k} most relevant papers to this idea. "
        "Return ONLY a valid JSON array of their 1-based indices, e.g. [1,3,5]. "
        "No explanation, no markdown, just the JSON array."
    )
    raw = vertex_generate(prompt, max_tokens=200, temperature=0.0)
    # Strip any accidental markdown fences
    raw = (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        indices = json.loads(raw)
        return [candidates[i - 1] for i in indices if 1 <= i <= len(candidates)][:top_k]
    except Exception:
        log.warning("Could not parse idea-filter response: %s", raw)
        return candidates[:top_k]


# ─── Non-Novel data point ───────────────────────────────────────────────────────


def build_non_novel_point(paper: dict, n_related: int = 10) -> Optional[dict]:
    pid = paper.get("paperId")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    if not abstract:
        return None

    idea = extract_idea(title, abstract)
    log.info("    idea: %s", idea[:120])

    sim_papers = s2_recommendations(pid, limit=20)
    cite_papers = get_citation_neighbours(
        paper, target_year=paper.get("year", 9999), limit=20
    )

    candidates = list(
        {p["paperId"]: p for p in sim_papers + cite_papers if p.get("paperId")}.values()
    )
    related = filter_related_by_idea(idea, candidates, top_k=n_related)

    if len(related) < 3:
        return None

    return {
        "paper_id": pid,
        "title": title,
        "idea": idea,
        "label": 0,
        "label_str": "Not Novel",
        "related_papers": [
            {
                "paper_id": p.get("paperId"),
                "title": p.get("title"),
                "abstract": p.get("abstract"),
            }
            for p in related
        ],
    }


# ─── Novel data point ───────────────────────────────────────────────────────────


def build_novel_point(paper: dict, n_related: int = 10) -> Optional[dict]:
    pid = paper.get("paperId")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    year = paper.get("year", 9999)

    if not abstract:
        return None

    idea = extract_idea(title, abstract)
    log.info("    idea: %s", idea[:120])

    # Only papers published BEFORE P
    cite_papers = get_citation_neighbours(paper, target_year=year, limit=30)
    sim_papers = [
        p
        for p in s2_recommendations(pid, limit=20)
        if p.get("year") and p["year"] < year
    ]

    candidates = list(
        {p["paperId"]: p for p in cite_papers + sim_papers if p.get("paperId")}.values()
    )
    related = filter_related_by_idea(idea, candidates, top_k=n_related)

    if len(related) < 3:
        return None

    return {
        "paper_id": pid,
        "title": title,
        "idea": idea,
        "label": 1,
        "label_str": "Novel",
        "related_papers": [
            {
                "paper_id": p.get("paperId"),
                "title": p.get("title"),
                "abstract": p.get("abstract"),
            }
            for p in related
        ],
    }


# ─── Seed paper fetching ────────────────────────────────────────────────────────


def fetch_novel_seeds(n: int) -> list[dict]:
    """Fetch high-quality papers from top venues as novel seeds."""
    seeds = []
    queries = [
        "transformer attention mechanism deep learning",
        "large language model pretraining",
        "graph neural network node classification",
        "diffusion model image generation",
        "contrastive self-supervised learning representation",
        "reinforcement learning policy gradient reward",
        "object detection convolutional neural network",
        "natural language processing question answering",
    ]
    for q in queries:
        if len(seeds) >= n:
            break
        for p in s2_search(q, limit=10):
            venue = (p.get("publicationVenue") or {}).get("name", "")
            if any(v in venue for v in TOP_VENUES) and p.get("abstract"):
                seeds.append(p)
            if len(seeds) >= n:
                break
        time.sleep(1)
    return seeds[:n]


def fetch_non_novel_seeds(n: int) -> list[dict]:
    """Fetch papers with likely high content overlap with existing work."""
    seeds = []
    queries = [
        "survey review overview deep learning methods",
        "extension improvement existing baseline model",
        "incremental update existing framework architecture",
        "modified version CNN image classification",
        "fine-tuning pretrained model downstream task",
    ]
    for q in queries:
        if len(seeds) >= n:
            break
        for p in s2_search(q, limit=10):
            if p.get("abstract"):
                seeds.append(p)
            if len(seeds) >= n:
                break
        time.sleep(1)
    return seeds[:n]


# ─── Main collector ─────────────────────────────────────────────────────────────


def collect_dataset(
    max_count: int = 20,
    n_related: int = 10,
    output_path: str = "dataset.jsonl",
    balance: bool = True,
) -> list[dict]:
    """
    Collect a dataset of `max_count` data points (balanced novel / non-novel).

    Args:
        max_count:   Total number of data points to collect.
        n_related:   Number of related papers per data point (R1..R10).
        output_path: Where to write the JSONL dataset.
        balance:     If True, aim for 50/50 novel vs non-novel split.

    Returns:
        List of dataset records.
    """
    half = max_count // 2 if balance else max_count

    log.info("Fetching novel seed papers …")
    novel_seeds = fetch_novel_seeds(half + 5)
    log.info("Fetching non-novel seed papers …")
    non_novel_seeds = fetch_non_novel_seeds(max_count - half + 5)

    dataset = []

    # ── Novel class ────────────────────────────────────────────────────────────
    log.info("Building Novel (1) data points …")
    novel_count = 0
    for paper in novel_seeds:
        if novel_count >= half:
            break
        log.info("  [Novel] %s", paper.get("title", "")[:80])
        point = build_novel_point(paper, n_related=n_related)
        if point:
            dataset.append(point)
            novel_count += 1
            log.info("  ✓ Novel collected (%d/%d)", novel_count, half)
        time.sleep(1.5)

    # ── Non-Novel class ────────────────────────────────────────────────────────
    log.info("Building Not-Novel (0) data points …")
    non_novel_target = max_count - len(dataset)
    non_novel_count = 0
    for paper in non_novel_seeds:
        if non_novel_count >= non_novel_target:
            break
        log.info("  [Non-Novel] %s", paper.get("title", "")[:80])
        point = build_non_novel_point(paper, n_related=n_related)
        if point:
            dataset.append(point)
            non_novel_count += 1
            log.info(
                "  ✓ Non-Novel collected (%d/%d)", non_novel_count, non_novel_target
            )
        time.sleep(1.5)

    # ── Write JSONL ────────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

    log.info(
        "Dataset saved → %s  |  total=%d  novel=%d  non-novel=%d",
        output_path,
        len(dataset),
        novel_count,
        non_novel_count,
    )
    return dataset


# ─── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect novel/non-novel paper dataset via Vertex AI"
    )
    p.add_argument("--max-count", type=int, default=20, help="Total data points")
    p.add_argument(
        "--n-related", type=int, default=10, help="Related papers per point (R1..R10)"
    )
    p.add_argument(
        "--output", type=str, default="dataset.jsonl", help="Output JSONL path"
    )
    p.add_argument("--no-balance", action="store_true", help="Don't balance classes")
    p.add_argument(
        "--s2-api-key", type=str, default="", help="Semantic Scholar API key"
    )
    # ── Vertex AI ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--vertex-project",
        type=str,
        default="",
        help="GCP project ID (or set VERTEX_PROJECT env var)",
    )
    p.add_argument(
        "--vertex-location",
        type=str,
        default="us-central1",
        help="Vertex AI region (default: us-central1)",
    )
    p.add_argument(
        "--vertex-model",
        type=str,
        default="gemini-2.5-pro",
        help="Vertex AI model name (default: gemini-2.5-pro)",
    )
    p.add_argument(
        "--vertex-api-key",
        type=str,
        default="",
        help="Path to GCP service-account JSON key file "
        "(sets GOOGLE_APPLICATION_CREDENTIALS automatically)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Apply CLI overrides ────────────────────────────────────────────────────
    if args.s2_api_key:
        S2_API_KEY = args.s2_api_key

    if args.vertex_api_key:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.vertex_api_key

    if args.vertex_project:
        VERTEX_PROJECT = args.vertex_project
    if args.vertex_location:
        VERTEX_LOCATION = args.vertex_location
    if args.vertex_model:
        VERTEX_MODEL = args.vertex_model

    if not VERTEX_PROJECT:
        raise SystemExit(
            "ERROR: GCP project required.\n"
            "  Pass --vertex-project YOUR_PROJECT_ID\n"
            "  or set the VERTEX_PROJECT environment variable."
        )

    collect_dataset(
        max_count=args.max_count,
        n_related=args.n_related,
        output_path=args.output,
        balance=not args.no_balance,
    )
