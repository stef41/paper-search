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
docker-compose up -d
```

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive Swagger docs.

## Testing

```bash
docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
```
