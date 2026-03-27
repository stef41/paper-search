#!/usr/bin/env python3
"""
Aggressive stress test simulating mass adoption of the ArXiv search API.

Scenarios:
  1. Sustained concurrent load (many agents querying simultaneously)
  2. Burst spike (sudden flood of requests)
  3. Heavy mixed workload (complex queries + simple lookups + stats)
  4. Deep pagination storm
  5. Regex and fuzzy barrage (expensive queries)
  6. Single-user rapid fire (one agent hammering)
  7. Write-read interleave (simulating live ingestion + reads)
  8. Connection exhaustion (max concurrent connections)
  9. Slow query avalanche (worst-case queries)
  10. Endurance run (sustained throughput over time)
"""

import os, sys, time, json, random, string, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

# Point directly at the live API to test real infra, not TestClient
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "changeme-key-1")

import urllib.request
import urllib.error


@dataclass
class Result:
    scenario: str
    total_requests: int
    successful: int
    failed: int
    errors_429: int
    errors_5xx: int
    errors_other: int
    latencies_ms: list[int] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def rps(self) -> float:
        return self.total_requests / self.elapsed_s if self.elapsed_s > 0 else 0

    @property
    def success_rate(self) -> float:
        return self.successful / self.total_requests * 100 if self.total_requests > 0 else 0

    @property
    def p50(self) -> int:
        return int(statistics.median(self.latencies_ms)) if self.latencies_ms else 0

    @property
    def p95(self) -> int:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def p99(self) -> int:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    @property
    def max_ms(self) -> int:
        return max(self.latencies_ms) if self.latencies_ms else 0


def api_call(method: str, path: str, body: dict | None = None) -> tuple[int, int]:
    """Makes an HTTP request. Returns (status_code, latency_ms)."""
    url = f"{API_URL}{path}"
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
            ms = int((time.monotonic() - t0) * 1000)
            return resp.status, ms
    except urllib.error.HTTPError as e:
        ms = int((time.monotonic() - t0) * 1000)
        return e.code, ms
    except Exception:
        ms = int((time.monotonic() - t0) * 1000)
        return 0, ms


def run_scenario(
    name: str,
    requests_fn,  # callable() -> list of (method, path, body) tuples
    concurrency: int,
    description: str = "",
) -> Result:
    tasks = requests_fn()
    total = len(tasks)
    result = Result(
        scenario=name,
        total_requests=total,
        successful=0, failed=0,
        errors_429=0, errors_5xx=0, errors_other=0,
    )

    print(f"\n{'━' * 60}")
    print(f"  SCENARIO: {name}")
    if description:
        print(f"  {description}")
    print(f"  Requests: {total} | Concurrency: {concurrency}")
    print(f"{'━' * 60}")

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(api_call, m, p, b) for m, p, b in tasks]
        for f in as_completed(futures):
            status, ms = f.result()
            result.latencies_ms.append(ms)
            if 200 <= status < 300:
                result.successful += 1
            elif status == 429:
                result.errors_429 += 1
                result.failed += 1
            elif status >= 500:
                result.errors_5xx += 1
                result.failed += 1
            else:
                result.errors_other += 1
                result.failed += 1
    result.elapsed_s = time.monotonic() - t0

    icon = "✓" if result.errors_5xx == 0 else "✗"
    print(f"  {icon} {result.successful}/{total} OK | "
          f"429s: {result.errors_429} | 5xx: {result.errors_5xx} | other: {result.errors_other}")
    print(f"    Throughput: {result.rps:.1f} req/s | Elapsed: {result.elapsed_s:.1f}s")
    print(f"    Latency p50={result.p50}ms p95={result.p95}ms p99={result.p99}ms max={result.max_ms}ms")
    return result


# ════════════════════════════════════════════════════════════
# QUERY GENERATORS
# ════════════════════════════════════════════════════════════

TOPICS = [
    "transformer", "attention mechanism", "graph neural network",
    "reinforcement learning", "generative adversarial", "variational autoencoder",
    "BERT", "GPT", "language model", "object detection", "image segmentation",
    "quantum computing", "topological", "string theory", "dark matter",
    "Ricci flow", "stochastic differential equations", "optimal transport",
    "federated learning", "differential privacy", "neural architecture search",
    "protein folding", "drug discovery", "climate modeling",
    "autonomous driving", "robotics control", "multi-agent",
    "knowledge graph", "recommender system", "speech recognition",
    "video understanding", "point cloud", "medical imaging",
    "time series forecasting", "anomaly detection", "causal inference",
]

AUTHORS = [
    "LeCun", "Hinton", "Bengio", "Vaswani", "Goodfellow",
    "Schmidhuber", "Sutton", "Silver", "Krizhevsky", "He",
    "Devlin", "Brown", "Radford", "Chen", "Li",
    "Wu", "Zhang", "Wang", "Liu", "Yang",
]

CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE",
    "cs.RO", "cs.CR", "stat.ML", "math.OC", "quant-ph",
    "hep-th", "astro-ph", "cond-mat", "physics.comp-ph",
]


def simple_queries(n: int):
    """Simple full-text search queries."""
    tasks = []
    for _ in range(n):
        topic = random.choice(TOPICS)
        tasks.append(("POST", "/search", {"query": topic, "limit": 20}))
    return tasks


def complex_queries(n: int):
    """Multi-filter combined queries."""
    tasks = []
    for _ in range(n):
        body: dict[str, Any] = {
            "query": random.choice(TOPICS),
            "categories": random.sample(CATEGORIES, k=random.randint(1, 3)),
            "sort_by": random.choice(["date", "relevance", "citations"]),
            "sort_order": random.choice(["asc", "desc"]),
            "limit": random.choice([10, 20, 50, 100]),
            "highlight": random.choice([True, False]),
        }
        if random.random() > 0.5:
            body["submitted_date"] = {"gte": f"{random.randint(2015, 2025)}-01-01T00:00:00+00:00"}
        if random.random() > 0.7:
            body["has_github"] = True
        if random.random() > 0.7:
            body["author"] = random.choice(AUTHORS)
        tasks.append(("POST", "/search", body))
    return tasks


def fuzzy_queries(n: int):
    """Fuzzy search with intentional typos."""
    typos = [
        "tansformer", "rienforcement lerning", "nueral neetwork",
        "convluton", "atention mecanism", "genrative adversrial",
        "languge modl", "objct detecton", "imge segmentaton",
        "autonmous drivng", "robtics contrl", "knowledg grph",
    ]
    tasks = []
    for _ in range(n):
        tasks.append(("POST", "/search", {
            "fuzzy": random.choice(typos),
            "fuzzy_fuzziness": random.choice([1, 2]),
            "limit": 20,
        }))
    return tasks


def regex_queries(n: int):
    """Regex searches (expensive on 3M docs)."""
    patterns = [
        ".*[Tt]ransformer.*", ".*BERT.*", ".*GAN.*",
        ".*[Nn]eural [Nn]etwork.*", ".*[Dd]eep [Ll]earning.*",
        ".*[Rr]einforcement.*", ".*[Qq]uantum.*", ".*GPT.*",
        ".*[Cc]onvolution.*", ".*[Aa]ttention.*",
    ]
    tasks = []
    for _ in range(n):
        tasks.append(("POST", "/search", {
            "title_regex": random.choice(patterns),
            "limit": 20,
        }))
    return tasks


def deep_pagination_queries(n: int):
    """Deep pagination at various offsets."""
    tasks = []
    for _ in range(n):
        offset = random.choice([100, 500, 1000, 2000, 5000, 8000, 9500])
        limit = random.choice([20, 50, 100])
        if offset + limit > 10000:
            limit = 10000 - offset
        tasks.append(("POST", "/search", {
            "sort_by": "date",
            "sort_order": "desc",
            "offset": offset,
            "limit": limit,
        }))
    return tasks


def stats_and_lookup_queries(n: int):
    """Mix of stats endpoint and paper lookups."""
    paper_ids = [
        "2301.00001", "2302.00001", "2303.00001", "2304.00001",
        "2305.00001", "2306.00001", "2307.00001", "2308.00001",
        "1706.03762", "hep-th/0508186",
    ]
    tasks = []
    for _ in range(n):
        if random.random() > 0.5:
            tasks.append(("GET", "/stats", None))
        else:
            pid = random.choice(paper_ids)
            tasks.append(("GET", f"/paper/{pid}", None))
    return tasks


def author_queries(n: int):
    """Nested author searches (expensive nested queries)."""
    tasks = []
    for _ in range(n):
        body: dict[str, Any] = {"author": random.choice(AUTHORS)}
        if random.random() > 0.5:
            body["categories"] = random.sample(CATEGORIES, k=random.randint(1, 2))
        if random.random() > 0.5:
            body["query"] = random.choice(TOPICS)
        tasks.append(("POST", "/search", body))
    return tasks


def title_abstract_queries(n: int):
    """Targeted title + abstract combined queries."""
    tasks = []
    for _ in range(n):
        body: dict[str, Any] = {}
        body["title_query"] = random.choice(TOPICS)
        body["abstract_query"] = random.choice(TOPICS)
        if random.random() > 0.5:
            body["operator"] = "and"
        tasks.append(("POST", "/search", body))
    return tasks


# ════════════════════════════════════════════════════════════
# SCENARIOS
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ARXIV SEARCH ENGINE — MASS ADOPTION STRESS TEST")
    print(f"  Target: {API_URL}")
    print(f"  Database: ~3M papers")
    print("=" * 60)

    # Verify API is alive
    try:
        status, ms = api_call("GET", "/health")
        if status != 200:
            print(f"  ✗ API not healthy (status={status}). Aborting.")
            sys.exit(1)
        print(f"  ✓ API healthy ({ms}ms)")
    except Exception as e:
        print(f"  ✗ Cannot reach API: {e}")
        sys.exit(1)

    results: list[Result] = []

    # ── 1. Sustained concurrent load: 50 agents querying for 500 requests ──
    results.append(run_scenario(
        "1. Sustained Load (50 concurrent agents)",
        lambda: simple_queries(500),
        concurrency=50,
        description="Simulates 50 agents making simple search queries simultaneously",
    ))

    # ── 2. Burst spike: 200 requests all at once ──
    results.append(run_scenario(
        "2. Burst Spike (200 simultaneous requests)",
        lambda: simple_queries(200),
        concurrency=200,
        description="Sudden flash of 200 concurrent requests hitting the API",
    ))

    # ── 3. Heavy mixed workload: 300 requests across all query types ──
    def mixed_workload():
        tasks = []
        tasks.extend(simple_queries(80))
        tasks.extend(complex_queries(60))
        tasks.extend(fuzzy_queries(40))
        tasks.extend(author_queries(40))
        tasks.extend(title_abstract_queries(30))
        tasks.extend(stats_and_lookup_queries(30))
        tasks.extend(deep_pagination_queries(20))
        random.shuffle(tasks)
        return tasks
    results.append(run_scenario(
        "3. Mixed Workload (300 diverse queries)",
        mixed_workload,
        concurrency=60,
        description="Realistic agent mix: simple, complex, fuzzy, author, stats, pagination",
    ))

    # ── 4. Deep pagination storm ──
    results.append(run_scenario(
        "4. Deep Pagination Storm (200 requests)",
        lambda: deep_pagination_queries(200),
        concurrency=40,
        description="Agents aggressively paginating through large result sets",
    ))

    # ── 5. Regex + fuzzy barrage (expensive queries) ──
    def expensive_queries():
        tasks = []
        tasks.extend(regex_queries(100))
        tasks.extend(fuzzy_queries(100))
        random.shuffle(tasks)
        return tasks
    results.append(run_scenario(
        "5. Expensive Query Barrage (200 regex+fuzzy)",
        expensive_queries,
        concurrency=40,
        description="Worst-case: regex scans + fuzzy matching on 3M docs concurrently",
    ))

    # ── 6. Rapid-fire single agent (sequential burst) ──
    results.append(run_scenario(
        "6. Rapid Fire (100 requests, sequential)",
        lambda: simple_queries(100),
        concurrency=1,
        description="Single agent hammering API as fast as possible",
    ))

    # ── 7. Author nested query storm ──
    results.append(run_scenario(
        "7. Nested Author Query Storm (200 requests)",
        lambda: author_queries(200),
        concurrency=50,
        description="Heavy nested Elasticsearch queries (author filter + category + text)",
    ))

    # ── 8. Connection exhaustion (max concurrency) ──
    results.append(run_scenario(
        "8. Connection Exhaustion (500 requests, 100 concurrent)",
        lambda: simple_queries(500),
        concurrency=100,
        description="Push connection pool to the limit with 100 concurrent connections",
    ))

    # ── 9. Worst-case combined queries ──
    def worst_case():
        tasks = []
        for _ in range(100):
            tasks.append(("POST", "/search", {
                "query": random.choice(TOPICS),
                "fuzzy": random.choice(["tansformer", "nueral neetwork", "convluton"]),
                "fuzzy_fuzziness": 2,
                "categories": random.sample(CATEGORIES, k=3),
                "submitted_date": {"gte": f"{random.randint(2015, 2025)}-01-01T00:00:00+00:00"},
                "has_github": True,
                "author": random.choice(AUTHORS),
                "sort_by": "date",
                "sort_order": "desc",
                "highlight": True,
                "limit": 100,
            }))
        return tasks
    results.append(run_scenario(
        "9. Worst-Case Combined (100 max-filter queries)",
        worst_case,
        concurrency=30,
        description="Every query uses text+fuzzy+category+date+github+author+sort+highlight+limit100",
    ))

    # ── 10. Endurance run (1000 requests sustained) ──
    def endurance():
        tasks = []
        tasks.extend(simple_queries(400))
        tasks.extend(complex_queries(300))
        tasks.extend(fuzzy_queries(150))
        tasks.extend(regex_queries(50))
        tasks.extend(stats_and_lookup_queries(100))
        random.shuffle(tasks)
        return tasks
    results.append(run_scenario(
        "10. Endurance Run (1000 requests sustained)",
        endurance,
        concurrency=80,
        description="Extended run simulating sustained mass adoption traffic",
    ))

    # ════════════════════════════════════════════════════
    # FINAL REPORT
    # ════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STRESS TEST FINAL REPORT")
    print("═" * 70)

    total_reqs = sum(r.total_requests for r in results)
    total_ok = sum(r.successful for r in results)
    total_429 = sum(r.errors_429 for r in results)
    total_5xx = sum(r.errors_5xx for r in results)
    total_other = sum(r.errors_other for r in results)
    total_elapsed = sum(r.elapsed_s for r in results)
    all_latencies = []
    for r in results:
        all_latencies.extend(r.latencies_ms)

    print(f"\n  Total requests:    {total_reqs}")
    print(f"  Successful:        {total_ok} ({total_ok/total_reqs*100:.1f}%)")
    print(f"  Rate limited (429):{total_429}")
    print(f"  Server errors (5xx):{total_5xx}")
    print(f"  Other errors:      {total_other}")
    print(f"  Total elapsed:     {total_elapsed:.1f}s")
    print(f"  Overall throughput:{total_reqs/total_elapsed:.1f} req/s")

    if all_latencies:
        s = sorted(all_latencies)
        print(f"\n  LATENCY DISTRIBUTION ({len(s)} samples):")
        print(f"    p50:  {s[len(s)//2]}ms")
        print(f"    p90:  {s[int(len(s)*0.9)]}ms")
        print(f"    p95:  {s[int(len(s)*0.95)]}ms")
        print(f"    p99:  {s[int(len(s)*0.99)]}ms")
        print(f"    max:  {s[-1]}ms")
        print(f"    mean: {statistics.mean(s):.0f}ms")

    print(f"\n  {'SCENARIO':<50} {'RPS':>6} {'p50':>6} {'p95':>6} {'p99':>6} {'max':>6} {'OK%':>6}")
    print(f"  {'─'*50} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for r in results:
        print(f"  {r.scenario:<50} {r.rps:>5.0f}  {r.p50:>5}  {r.p95:>5}  {r.p99:>5}  {r.max_ms:>5}  {r.success_rate:>5.1f}")

    # Verdict
    print()
    critical_failures = total_5xx > 0
    high_latency = any(r.p99 > 10000 for r in results)
    low_throughput = any(r.rps < 5 for r in results if r.total_requests > 50)

    if critical_failures:
        print("  🔴 VERDICT: FAIL — Server errors under load (5xx responses)")
    elif high_latency:
        print("  🟡 VERDICT: DEGRADED — p99 latency exceeded 10s in some scenarios")
    elif low_throughput:
        print("  🟡 VERDICT: DEGRADED — Throughput dropped below 5 req/s")
    elif total_429 > total_reqs * 0.2:
        print("  🟡 VERDICT: RATE-LIMITED — Over 20% of requests were throttled")
    else:
        print("  🟢 VERDICT: PASS — API handled mass adoption load successfully")

    print()
    sys.exit(1 if critical_failures else 0)


if __name__ == "__main__":
    main()
