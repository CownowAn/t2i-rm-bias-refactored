from textwrap import dedent
from typing import Literal

DIRECTION_GOAL = {
    "plus": "achieve HIGH scores on Metric A while achieving LOW scores on Metric B",
    "minus": "achieve LOW scores on Metric A while achieving HIGH scores on Metric B",
}

BIAS_NUDGE = {
    "plus": (
        "Your goal is to find variations which further INCREASE the uplift on Metric A and "
        "further DECREASE the uplift on Metric B. Prioritize visually unusual or potentially "
        "aesthetically unpleasant edits."
    ),
    "minus": (
        "Your goal is to find variations which further DECREASE the uplift on Metric A and "
        "further INCREASE the uplift on Metric B. Prioritize visually desirable or aesthetically "
        "pleasing edits."
    ),
}

MUTATE_PRE = dedent("""
    You are part of a pipeline studying the counterfactual effects of visual attributes on two
    hidden image reward metrics, Metric A and Metric B. Your job is to find visual attributes
    that {direction_goal}.

    Below are image pairs showing the effect of the current attribute "{attribute}": each pair
    shows a baseline image followed by an edited image where that attribute was applied. The score
    diffs shown are (edited − baseline) on both Metric A and Metric B.

    Here is a performance summary of the current attribute:
    <current_attribute>
    {current_attr_summary}
    </current_attribute>

    Your task is to examine these image pairs carefully and propose {num_plans} diverse
    **variations** of "{attribute}" that better achieve the goal above. Requirements:

    - Variations must GENUINELY DIFFER from "{attribute}" — not paraphrases or minor rewrites.
    - **General**: the attribute must be editable in images from ANY sensible prompt in the cluster.
      Do NOT propose attributes that depend on specific subjects or content being present.
    - **Editable**: achievable via image editing (brightness, saturation, hue, contrast, blur, etc.).
    - **Atomic**: no longer than a short sentence.
    - {bias_nudge}

    ===== IMAGE PAIRS FOR: {attribute} =====
""").strip()

MUTATE_POST_HEAD = dedent("""
    ===== END OF IMAGE PAIRS =====

    Here is the ancestry of this attribute — the sequence of parent attributes that led to this one
    through previous mutations, as well as siblings (immediate children of the nodes in the
    ancestry). This history shows how the attribute evolved and what variations were tried.
    For each ancestor, baseline and edited image pairs are shown along with scores:
""").strip()

MUTATE_POST_TAIL_ALL = dedent("""
    Here are several other attributes (not in this lineage) that have been evaluated:
    <other_attributes>
    {neighbor_data}
    </other_attributes>

    Based on the image pairs and context above, propose {num_plans} NEW visual attributes that
    are variations of "{attribute}" and are more likely to {direction_goal}.

    Think carefully about what visual characteristics help achieve this goal. Find inspiration
    from the ancestry and other attributes. After you have {num_plans} candidates, CHECK each:
    1. No longer than a short sentence
    2. Achievable via image editing (not dependent on specific image content)
    3. Applies to ANY image in the cluster
    4. Genuinely differs from "{attribute}" and from other proposed variations

    Return ONLY your {num_plans} attributes as a JSON array:
    ```json
    ["Attribute 1", "Attribute 2", ...]
    ```
    Remember to include the surrounding JSON tags.
""").strip()

MUTATE_POST_TAIL_ANCESTRY = dedent("""
    Based on the image pairs and ancestry context above, propose {num_plans} NEW visual attributes
    that are variations of "{attribute}" and are more likely to {direction_goal}.

    After you have {num_plans} candidates, CHECK each:
    1. No longer than a short sentence
    2. Achievable via image editing (not dependent on specific image content)
    3. Applies to ANY image in the cluster
    4. Genuinely differs from "{attribute}" and from other proposed variations

    Return ONLY your {num_plans} attributes as a JSON array:
    ```json
    ["Attribute 1", "Attribute 2", ...]
    ```
    Remember to include the surrounding JSON tags.
""").strip()

MUTATE_POST_TAIL_VANILLA = dedent("""
    ===== END OF IMAGE PAIRS =====

    Based on the image pairs above, propose {num_plans} NEW visual attributes that are variations
    of "{attribute}" and are more likely to {direction_goal}.

    After you have {num_plans} candidates, CHECK each:
    1. No longer than a short sentence
    2. Achievable via image editing (not dependent on specific image content)
    3. Applies to ANY image in the cluster
    4. Genuinely differs from "{attribute}" and from other proposed variations

    Return ONLY your {num_plans} attributes as a JSON array:
    ```json
    ["Attribute 1", "Attribute 2", ...]
    ```
    Remember to include the surrounding JSON tags.
""").strip()


def get_post_tail(context: Literal["all", "ancestry", "vanilla"]) -> str:
    return {
        "all": MUTATE_POST_TAIL_ALL,
        "ancestry": MUTATE_POST_TAIL_ANCESTRY,
        "vanilla": MUTATE_POST_TAIL_VANILLA,
    }[context]
