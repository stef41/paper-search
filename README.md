# ArXiv Paper Search — Copilot Skill

A GitHub Copilot skill that gives agents access to a live ArXiv paper database (~3M papers, all categories, 2005–present).

## What's in this repo

```
.github/
  agents/arxiv.agent.md      # Agent definition for the @arxiv agent
  skills/paper-search/SKILL.md  # Skill with full API docs, schemas, examples
```

## Capabilities

- **Full-text search** across titles, abstracts, and authors
- **Semantic similarity** (sentence-transformer embeddings, boost/exclude modes)
- **Fuzzy & regex** search on all text fields
- **Filters**: categories, dates, citations, h-index, page count, GitHub availability
- **55 graph algorithms**: co-authorship, citation networks, PageRank, community detection, shortest paths, centrality metrics, pattern matching, and more
- **Pipeline composition**: chain graph algorithms into multi-step workflows

## Usage

Install this skill by adding the repo to your Copilot workspace, then:

- Use `@arxiv` to invoke the agent directly
- Or let Copilot auto-invoke the `paper-search` skill when you ask about research papers

### Example prompts

```
@arxiv find recent papers on diffusion models in cs.AI with >50 citations
@arxiv who are the top co-authors of Yann LeCun?
@arxiv show the citation network around attention is all you need
@arxiv find papers similar to 2401.12345
```
