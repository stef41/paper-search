# ArXiv Search Engine for Agents

A high-performance, always up-to-date ArXiv search engine with advanced query capabilities designed for AI agents.

## Features

- **Full ArXiv Database**: Continuously synced via OAI-PMH and bulk data
- **Multi-Level Semantic Similarity**: Search by title, abstract, or paragraph-level embeddings
- **Advanced Filtering**: Author h-index, citation count, first-author, median h-index
- **Citation Analytics**: Statistics on citing and cited papers
- **Fuzzy Matching**: Approximate string matching on all text fields
- **Regex Search**: Full regex support on titles, abstracts, authors
- **Boolean Queries**: Complex AND/OR/NOT query composition
- **Minimum-Should-Match**: Control how many optional clauses must match
- **Date Filtering**: Filter by submission, update, or publication date
- **Paper Length**: Filter by page count
- **Subcategory Filtering**: ArXiv category/subcategory hierarchy
- **GitHub Detection**: Filter papers that have associated GitHub repos
- **Sort By**: Relevance, date, citations, h-index, and more
- **Rate Limiting**: Configurable per-client rate limits
- **Security First**: Input validation, sandboxed execution, no injection vectors

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Agents    │────▶│   FastAPI    │────▶│  Elasticsearch   │
│  (clients)  │◀────│   Gateway    │◀────│  (search+store)  │
└─────────────┘     └──────┬───────┘     └──────────────────┘
                           │                      ▲
                    ┌──────▼───────┐              │
                    │   Redis      │     ┌────────┴─────────┐
                    │ (cache+rate) │     │   Ingestion       │
                    └──────────────┘     │   Worker          │
                                        └──────────────────┘
```

## Quick Start

```bash
cp .env.example .env

# Start infrastructure
docker-compose up -d elasticsearch redis api

# Seed initial data (~40,000 papers from 20 ArXiv categories)
docker-compose run --rm seed

# Or seed faster without embeddings, add them later:
docker-compose run --rm seed python -m src.ingestion.seed --max-papers 5000 --skip-embeddings
docker-compose run --rm seed python -m src.ingestion.embed_backfill

# Start continuous OAI-PMH harvester for ongoing updates
docker-compose up -d ingestion

# Enrich papers with Semantic Scholar citation data
docker-compose --profile enrich run --rm enrich
```

### Full Database Import (2.4M papers)

Download the [Kaggle ArXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv):

```bash
# Place the JSON file in ./data/ directory
docker-compose run --rm seed python -m src.ingestion.bulk_import \
    --file /data/arxiv-metadata-oai-snapshot.json \
    --skip-embeddings \
    --batch-size 1000

# Then backfill embeddings in background
docker-compose run --rm seed python -m src.ingestion.embed_backfill
```

### Ingestion Commands

| Command | Purpose |
|---------|---------|
| `src.ingestion.seed` | Fast initial load via ArXiv API (~200 papers/min) |
| `src.ingestion.bulk_import` | Import Kaggle JSON snapshot (2.4M papers) |
| `src.ingestion.worker` | Continuous OAI-PMH harvesting with state tracking |
| `src.ingestion.enrich` | Add citation/h-index data from Semantic Scholar |
| `src.ingestion.embed_backfill` | Generate embeddings for papers imported without them |

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive Swagger docs.

## Testing

```bash
docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
```
