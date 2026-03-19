import logging
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_DATE = datetime(2023, 10, 3)  # October 3, 2023
MIN_ABSTRACT_LENGTH = 20
DEFAULT_EMBEDDING_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Keyword Extraction
# ---------------------------------------------------------------------------


def extract_keywords(topic: str, num_keywords: int = 5) -> list[str]:
    """
    Extract keywords from a research topic.

    Uses a simple approach: splits topic into words, removes common stop words,
    and returns the top keywords by frequency and position.

    Args:
        topic: The research topic/title as a string
        num_keywords: Number of keywords to extract

    Returns:
        List of extracted keywords
    """
    # Common English stop words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
    }

    # Clean and tokenize
    tokens = topic.lower().split()
    keywords = [
        token.strip(".,;!?()[]{}\"':")
        for token in tokens
        if token.lower().strip(".,;!?()[]{}\"':") not in stop_words
        and len(token.strip(".,;!?()[]{}\"':")) > 2
    ]

    # Return top unique keywords
    return list(dict.fromkeys(keywords))[:num_keywords]


# ---------------------------------------------------------------------------
# Paper Retrieval
# ---------------------------------------------------------------------------


def retrieve_papers_for_keywords(
    keywords: list[str],
    s2_client: Any,  # S2Client instance
    papers_per_keyword: int = 10,
) -> list[dict]:
    """
    Retrieve papers from Semantic Scholar API for given keywords.

    Args:
        keywords: List of keywords to search for
        s2_client: Initialized S2Client instance
        papers_per_keyword: Number of papers to retrieve per keyword

    Returns:
        List of paper dictionaries with deduplicated results
    """
    papers = {}  # Use dict for deduplication by paperId

    for keyword in keywords:
        logger.info(f"Retrieving papers for keyword: {keyword}")
        try:
            results = s2_client.search_papers(query=keyword, limit=papers_per_keyword)

            for paper in results:
                if paper and "paperId" in paper:
                    paper_id = paper["paperId"]
                    if paper_id not in papers:
                        papers[paper_id] = paper

        except Exception as e:
            logger.warning(f"Error retrieving papers for '{keyword}': {e}")

    paper_list = list(papers.values())
    logger.info(f"Retrieved {len(paper_list)} unique papers")
    return paper_list


# ---------------------------------------------------------------------------
# Embedding Generation
# ---------------------------------------------------------------------------


def generate_embeddings(
    texts: list[str], model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Generate sentence embeddings for a list of texts.

    Uses sentence-transformers for fast, efficient embedding generation.

    Args:
        texts: List of text strings to embed
        model_name: Name of the sentence-transformers model to use

    Returns:
        NumPy array of shape (len(texts), embedding_dim)
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Filter out empty texts
    valid_texts = [t for t in texts if t and len(t.strip()) > 0]

    if not valid_texts:
        logger.warning("No valid texts for embedding generation")
        return np.array([])

    logger.info(f"Generating embeddings for {len(valid_texts)} texts")
    embeddings = model.encode(
        valid_texts,
        batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return embeddings


# ---------------------------------------------------------------------------
# Distance Computation
# ---------------------------------------------------------------------------


def compute_semantic_distance(
    idea_embedding: np.ndarray, paper_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine distance between an idea embedding and multiple paper embeddings.

    Args:
        idea_embedding: 1D array of shape (embedding_dim,)
        paper_embeddings: 2D array of shape (num_papers, embedding_dim)

    Returns:
        1D array of cosine distances of shape (num_papers,)
    """
    if paper_embeddings.size == 0:
        return np.array([])

    # Reshape idea_embedding to 2D array for cosine_distances
    idea_reshaped = idea_embedding.reshape(1, -1)
    distances = cosine_distances(idea_reshaped, paper_embeddings)[0]

    return distances


def compute_mean_distance(distances: np.ndarray) -> float:
    """
    Compute mean distance from an array of distances.

    Args:
        distances: 1D array of distance values

    Returns:
        Mean distance, or 0.0 if array is empty
    """
    if distances.size == 0:
        return 0.0
    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# Paper Classification
# ---------------------------------------------------------------------------


def classify_papers(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Classify papers into historical and contemporary groups based on publication date.

    Reference date: October 3, 2023
    - Historical: papers published before the reference date
    - Contemporary: papers published after the reference date

    Args:
        papers: List of paper dictionaries from Semantic Scholar API

    Returns:
        Tuple of (historical_papers, contemporary_papers)
    """
    historical = []
    contemporary = []

    for paper in papers:
        year = paper.get("year")

        if year is None:
            logger.debug(f"Paper {paper.get('paperId')} has no year, skipping")
            continue

        # Treat year as publication date (Jan 1 of that year for simplicity)
        paper_date = datetime(year, 1, 1)

        if paper_date < REFERENCE_DATE:
            historical.append(paper)
        else:
            contemporary.append(paper)

    logger.info(
        f"Classified papers: {len(historical)} historical, {len(contemporary)} contemporary"
    )
    return historical, contemporary


# ---------------------------------------------------------------------------
# Idea Text Generation
# ---------------------------------------------------------------------------


def generate_idea_text(paper: dict) -> str:
    """
    Generate a text representation of an idea from paper metadata.

    Uses title and abstract if available, falls back to title only.

    Args:
        paper: Paper dictionary from Semantic Scholar API

    Returns:
        String representation of the paper/idea
    """
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()

    # Combine title and abstract, with validation
    if abstract and len(abstract) > MIN_ABSTRACT_LENGTH:
        return f"{title}. {abstract}"
    else:
        return title


def generate_idea_texts(papers: list[dict]) -> list[str]:
    """
    Generate text representations for multiple papers/ideas.

    Args:
        papers: List of paper dictionaries

    Returns:
        List of idea text strings
    """
    texts = []
    for paper in papers:
        text = generate_idea_text(paper)
        if text.strip():
            texts.append(text)

    return texts


# ---------------------------------------------------------------------------
# Score Computation
# ---------------------------------------------------------------------------


def compute_novelty_score(
    generated_idea: str,
    papers: list[dict],
    s2_client: Any = None,
    keywords: list[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    Compute the novelty score for a generated research idea.

    This is the main function that orchestrates the entire novelty assessment process.

    Novelty Score Formula:
        SN = (1 + cd_ideas) * cc / (1 + hd_ideas)

    Args:
        generated_idea: The text description of the generated idea (title + description)
        papers: List of paper dictionaries from Semantic Scholar API.
                If None and s2_client is provided, papers will be retrieved using keywords.
        s2_client: Optional S2Client instance for retrieving papers if not provided
        keywords: Optional list of keywords for paper retrieval (auto-extracted if None)
        embedding_model: Name of sentence-transformers model to use

    Returns:
        Dictionary containing:
            - novelty_score: The computed SN value
            - historical_distance: Mean distance to historical papers (hd_ideas)
            - contemporary_distance: Mean distance to contemporary papers (cd_ideas)
            - contemporary_citations: Mean citation count of contemporary papers (cc)
            - num_historical_papers: Count of historical papers
            - num_contemporary_papers: Count of contemporary papers
            - reference_date: The date used to split historical/contemporary
            - error: Error message if computation failed (optional)
    """
    result = {
        "reference_date": REFERENCE_DATE.isoformat(),
    }

    try:
        # Validate input
        if not generated_idea or not generated_idea.strip():
            raise ValueError("Generated idea text cannot be empty")

        # Retrieve papers if needed
        if papers is None:
            if s2_client is None:
                raise ValueError(
                    "Either 'papers' must be provided or 's2_client' must be initialized"
                )

            if keywords is None:
                keywords = extract_keywords(generated_idea)
                logger.info(f"Extracted keywords: {keywords}")

            papers = retrieve_papers_for_keywords(keywords, s2_client)

            if not papers:
                raise ValueError("No papers retrieved from Semantic Scholar API")

        # Classify papers
        historical_papers, contemporary_papers = classify_papers(papers)

        result["num_historical_papers"] = len(historical_papers)
        result["num_contemporary_papers"] = len(contemporary_papers)

        # Validate paper counts
        if len(historical_papers) == 0:
            logger.warning("No historical papers found")
        if len(contemporary_papers) == 0:
            logger.warning("No contemporary papers found")

        # Generate embeddings
        logger.info("Generating embeddings...")

        # Embedding for the generated idea
        idea_embedding = generate_embeddings(
            [generated_idea], model_name=embedding_model
        )[0]

        # Embeddings for historical papers
        historical_texts = generate_idea_texts(historical_papers)
        historical_embeddings = (
            generate_embeddings(historical_texts, model_name=embedding_model)
            if historical_texts
            else np.array([])
        )

        # Embeddings for contemporary papers
        contemporary_texts = generate_idea_texts(contemporary_papers)
        contemporary_embeddings = (
            generate_embeddings(contemporary_texts, model_name=embedding_model)
            if contemporary_texts
            else np.array([])
        )

        # Compute distances
        logger.info("Computing semantic distances...")

        historical_distances = compute_semantic_distance(
            idea_embedding, historical_embeddings
        )
        hd_ideas = compute_mean_distance(historical_distances)

        contemporary_distances = compute_semantic_distance(
            idea_embedding, contemporary_embeddings
        )
        cd_ideas = compute_mean_distance(contemporary_distances)

        # Compute citation statistics
        contemporary_citations = [
            p.get("citationCount", 0) for p in contemporary_papers
        ]
        cc = float(np.mean(contemporary_citations)) if contemporary_citations else 0.0

        # Compute novelty score
        denominator = 1 + hd_ideas
        numerator = (1 + cd_ideas) * cc
        novelty_score = numerator / denominator if denominator > 0 else 0.0

        # Populate result
        result.update(
            {
                "novelty_score": float(novelty_score),
                "historical_distance": float(hd_ideas),
                "contemporary_distance": float(cd_ideas),
                "contemporary_citations": float(cc),
            }
        )

        logger.info(f"Novelty score computed: {novelty_score:.4f}")

    except Exception as e:
        logger.error(f"Error computing novelty score: {e}")
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------


def compute_novelty_scores_batch(
    ideas: list[dict],
    papers_per_idea: list[list[dict]] = None,
    s2_client: Any = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """
    Compute novelty scores for multiple generated ideas.

    Args:
        ideas: List of dictionaries, each containing:
            - "text": the idea text
            - "keywords": optional list of keywords
        papers_per_idea: Optional list matching ideas, each containing paper lists
        s2_client: Optional S2Client for paper retrieval
        embedding_model: Name of sentence-transformers model to use

    Returns:
        List of result dictionaries from compute_novelty_score()
    """
    results = []

    for idx, idea in enumerate(ideas):
        idea_text = idea.get("text", "")
        keywords = idea.get("keywords")
        papers = papers_per_idea[idx] if papers_per_idea else None

        logger.info(f"Processing idea {idx + 1}/{len(ideas)}")

        score_result = compute_novelty_score(
            generated_idea=idea_text,
            papers=papers,
            s2_client=s2_client,
            keywords=keywords,
            embedding_model=embedding_model,
        )

        score_result["idea_index"] = idx
        results.append(score_result)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    generated_idea = (
        "A retrieval-augmented framework for autonomous scientific hypothesis generation "
        "using multimodal literature signals"
    )

    # Minimal sample papers. Years are used to split historical vs contemporary.
    sample_papers = [
        {
            "paperId": "p1",
            "title": "Neural Methods for Scientific Discovery",
            "abstract": "We study language model guidance for hypothesis discovery in biomedicine.",
            "year": 2021,
            "citationCount": 120,
        },
        {
            "paperId": "p2",
            "title": "Literature-Informed AI for Research Ideation",
            "abstract": "This work combines retrieval and planning for automated idea generation.",
            "year": 2024,
            "citationCount": 35,
        },
        {
            "paperId": "p3",
            "title": "Multimodal Signals for Scientific Trend Forecasting",
            "abstract": "We use text and citation graphs to predict emerging topics.",
            "year": 2025,
            "citationCount": 18,
        },
    ]

    result = compute_novelty_score(
        generated_idea=generated_idea,
        papers=sample_papers,
        embedding_model="all-MiniLM-L6-v2",
    )

    print("Novelty score example result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
