"""Attribute clustering: deduplicate semantically similar attributes between evo steps."""
from __future__ import annotations
import json
from collections import defaultdict
from textwrap import dedent
from typing import Any

import numpy as np
from loguru import logger

from caller import AutoCaller
from search.utils.io import parse_json_response


# ─── Prompts ──────────────────────────────────────────────────────────────────

_CLUSTER_PROMPT = dedent("""
    You will be given a list of visual attribute descriptions.

    Your task is to cluster these attributes into clusters. Go through the list from top to bottom,
    maintaining a running list of clusters.

    - If the new attribute is semantically similar to a previous cluster (i.e., it would manifest
      almost identically in actual images), add it to that cluster.
    - If it is NOT similar to any previous cluster, create a new cluster with this attribute.

    After clustering, pick the most representative attribute from each cluster. Return your result:

    ```json
    [
        {{
            "representative": {{"index": ..., "attribute": ...}},
            "members": [{{"index": ..., "attribute": ...}}, ...]
        }},
        ...
    ]
    ```

    Here is the full list:
    {attributes}
""").strip()


# ─── Embed-based deduplication ────────────────────────────────────────────────

class EmbedDeduplicator:
    """Fast cosine-similarity deduplication using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def deduplicate(
        self,
        attributes: list[str],
        *,
        cosine_sim_threshold: float = 0.9,
        n_pop: int | None = None,
    ) -> list[str]:
        """Return up to n_pop representative attributes after deduplication."""
        if not attributes:
            return []

        embs: np.ndarray = self._model.encode(attributes, normalize_embeddings=True)
        sim = embs @ embs.T  # [N, N]

        # Union-Find connected components
        parent = list(range(len(attributes)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        rows, cols = np.where(np.triu(sim, k=1) >= cosine_sim_threshold)
        for i, j in zip(rows.tolist(), cols.tolist()):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        # Build components → pick medoid
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(attributes)):
            groups[find(i)].append(i)

        reps: list[str] = []
        for members in groups.values():
            if len(members) == 1:
                reps.append(attributes[members[0]])
            else:
                sub = embs[members]
                dists = 1.0 - sub @ sub.T
                medoid = members[int(np.argmin(dists.sum(axis=1)))]
                reps.append(attributes[medoid])

        if n_pop is not None:
            reps = reps[:n_pop]

        logger.info(f"Deduplicated {len(attributes)} → {len(reps)} attributes (threshold={cosine_sim_threshold})")
        return reps


# ─── LLM-based clustering ─────────────────────────────────────────────────────

class AttributeClusterer:
    """LLM-based attribute clusterer for inter-step deduplication."""

    def __init__(
        self,
        model_name: str = "openai/gpt-5.2",
        max_tokens: int = 50000,
        reasoning: str | None = "high",
        max_parallel: int = 64,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.max_parallel = max_parallel
        self.caller = AutoCaller(dotenv_path=".env")

    async def cluster(
        self,
        attributes: list[str],
        *,
        cluster_summary: str,
        n_pop: int,
    ) -> list[str]:
        """Cluster a list of attributes and return up to n_pop representatives."""
        if not attributes:
            return []
        if len(attributes) <= n_pop:
            return attributes

        prompt = _CLUSTER_PROMPT.format(
            cluster_summary=cluster_summary,
            attributes=json.dumps([{"index": i, "attribute": a} for i, a in enumerate(attributes)]),
        )

        responses = await self.caller.call(
            messages=[prompt],
            model=self.model_name,
            max_parallel=1,
            max_tokens=self.max_tokens,
            reasoning=self.reasoning,
            enable_cache=False,
            desc="Clustering attributes",
        )

        if not responses or responses[0] is None:
            logger.warning("Cluster LLM returned None; returning first n_pop attributes")
            return attributes[:n_pop]

        cluster_results, reasoning = parse_json_response(responses[0])
        logger.info(f"Clustering reasoning:\n{reasoning}")

        if not isinstance(cluster_results, list):
            logger.warning("Cluster result not a list; returning first n_pop")
            return attributes[:n_pop]

        # Sort by cluster size, take top n_pop
        try:
            cluster_results.sort(key=lambda x: len(x.get("members", [])), reverse=True)
        except Exception as e:
            logger.warning(f"Sorting cluster results failed: {e}")

        reps: list[str] = []
        for cluster in cluster_results[:n_pop]:
            rep = cluster.get("representative", {})
            idx = rep.get("index")
            if idx is not None and 0 <= idx < len(attributes):
                reps.append(attributes[idx])
            else:
                reps.append(rep.get("attribute", ""))

        logger.info(f"Clustered {len(attributes)} → {len(reps)} attributes")
        return [r for r in reps if r]

    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "reasoning": self.reasoning}
