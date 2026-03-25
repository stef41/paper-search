"""Quick seed script for local testing."""
import asyncio
import sys
import os

# Force localhost connections
os.environ["ES_HOST"] = "localhost"
os.environ["REDIS_HOST"] = "localhost"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import reset_settings
reset_settings()

from src.ingestion.seed import run_seed

async def main():
    cats = sys.argv[1:] if len(sys.argv) > 1 else [
        "cs.AI", "cs.CL", "cs.LG", "cs.CV", "cs.NE",
        "cs.IR", "cs.SE", "cs.RO", "cs.CR",
        "stat.ML", "quant-ph",
    ]
    result = await run_seed(
        categories=cats,
        max_papers_per_category=200,
        skip_embeddings=True,
    )
    print(f"Total seeded: {result}")

if __name__ == "__main__":
    asyncio.run(main())
