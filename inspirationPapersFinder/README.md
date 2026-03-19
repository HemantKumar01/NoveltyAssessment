# related_research_finder

Find papers related to a target research paper using the
[Semantic Scholar Academic Graph API](https://api.semanticscholar.org/api-docs/).

Two complementary sources are queried and merged:

| Source | Method | What it returns |
|---|---|---|
| **References** | `GET /paper/{id}/references` | Papers that the target paper cites |
| **Recommendations** | `POST /recommendations/v1/papers` | Papers S2's ML model considers topically similar |

An optional third source can be enabled:

| Source | Method | What it returns |
|---|---|---|
| **Citations** (opt-in) | `GET /paper/{id}/citations` | Papers that cite the target |

Results are deduplicated across sources and sorted by citation count.

---

## Setup

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

---

## Usage

```
python main.py <paper> [options]
```

### Paper identifier

The `<paper>` argument accepts any of the following forms:

| Format | Example |
|---|---|
| Semantic Scholar paper ID | `204e3073870fae3d05bcbc2f6a8e263d9b72e776` |
| arXiv ID | `arXiv:1706.03762` |
| DOI | `DOI:10.18653/v1/N18-1202` |
| Corpus ID | `CorpusID:13756489` |
| Free-text title | `"Attention is all you need"` |

### Options

| Flag | Default | Description |
|---|---|---|
| `--max-references N` | 200 | Max cited papers to retrieve |
| `--max-recommendations N` | 100 | Max S2-recommended papers |
| `--include-citations` | off | Also include papers that cite the target |
| `--max-citations N` | 100 | Max citing papers (only with `--include-citations`) |
| `--top N` | 20 | Papers shown in the CLI summary |
| `-o / --output FILE` | — | Save full results as JSON |
| `--api-key KEY` | — | S2 API key (optional; raises rate limit) |
| `-v / --verbose` | off | Show debug logging |

### Examples

```bash
# arXiv ID, save full JSON output
python main.py "arXiv:1706.03762" -o results.json

# DOI, include incoming citations, show top 30
python main.py "DOI:10.18653/v1/N18-1202" --include-citations --top 30

# Free-text title
python main.py "BERT pre-training of deep bidirectional transformers"

# With an API key for higher rate limits
python main.py "arXiv:1706.03762" --api-key YOUR_S2_API_KEY
```

---


**JSON file** (with `-o results.json`):

```json
{
  "targetPaper": { "paperId": "...", "title": "...", ... },
  "relatedPapers": [
    {
      "paperId": "...",
      "title": "...",
      "year": 2017,
      "authors": ["..."],
      "venue": "...",
      "citationCount": 12345,
      "abstract": "...",
      "url": "https://...",
      "pdfUrl": "https://...",
      "externalIds": { "ArXiv": "...", "DOI": "..." },
      "sources": ["reference", "recommendation"]
    }
  ],
  "stats": {
    "total": 230,
    "fromReferences": 130,
    "fromRecommendations": 100,
    "fromCitations": 0,
    "foundByMultipleSources": 14
  }
}
```

---

## File structure

```
related_research_finder/
├── s2_client.py   # Semantic Scholar API client (rate limiting, retries)
├── finder.py      # Core logic: fetch, merge, deduplicate related papers
├── main.py        # CLI entry point
└── requirements.txt
```

---

## API key

Without an API key the Semantic Scholar API allows ~1 request/second.
You can request a free key at <https://www.semanticscholar.org/product/api#api-key-form>
and pass it with `--api-key`.
