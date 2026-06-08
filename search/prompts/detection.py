from textwrap import dedent

ATTRIBUTE_DETECTION_SYSTEM = (
    "You are an expert visual analyst. Your task is to determine whether a specific visual "
    "attribute is present in the provided image."
)

_DETECTION_HEADER = dedent("""\
    Examine the provided image and determine whether it exhibits the following visual attribute.

    Attribute to detect: {attribute}

    """)

_DETECTION_PROMPT_BLOCK = dedent("""\
    The image was generated from this text prompt:
    <prompt>
    {prompt}
    </prompt>

    """)

_DETECTION_FOOTER_WITH_REASONING = dedent("""\
    Focus purely on whether the specific visual attribute is visually present in the image.
    Do NOT judge overall quality or preference.

    Respond in the following JSON format:

    ```json
    {{
        "present": <true|false>,
        "confidence": <0.0 to 1.0>,
        "reasoning": "Brief explanation"
    }}
    ```

    Respond with only the JSON block.""")

_DETECTION_FOOTER_NO_REASONING = dedent("""\
    Focus purely on whether the specific visual attribute is visually present in the image.
    Do NOT judge overall quality or preference.

    Respond in the following JSON format:

    ```json
    {{
        "present": <true|false>,
        "confidence": <0.0 to 1.0>
    }}
    ```

    Respond with only the JSON block.""")


# ── Applicability variants ────────────────────────────────────────────────────
# These ask the model to ALSO decide whether the image's content even makes
# the attribute evaluable. E.g. attribute talks about rocks but the image
# contains no rocks → applicable=false. Downstream code maps applicable=false
# to -1 (vs 0/1 for not-present/present) so callers can distinguish "doesn't
# have it" from "doesn't even apply".

_DETECTION_FOOTER_APPLICABILITY_WITH_REASONING = dedent("""\
    You must evaluate the given attribute in two strict, sequential steps:

    Step 1: Check Object Existence (Applicability)
    - Decide whether the core object/context described in the attribute exists in the image. (e.g., If the attribute is about "how rocks are rendered", simply check if "rocks" exist).
    - If the object does NOT exist, the attribute is NOT applicable. Immediately set "applicable": false and "present": false. Do not look for the attribute itself.
    - If the object exists, set "applicable": true and proceed ONLY to Step 2.

    Step 2: Check Visual Presence (Only if Step 1 is true)
    - Only when "applicable" is true, evaluate whether the specific attribute is visually present on that object.
    - Focus purely on visual presence; do NOT judge overall quality or preference.

    Respond in the following JSON format:
    {{
        "applicable": <true|false>,
        "present": <true|false>,
        "confidence": <0.0 to 1.0>,
        "reasoning": "Brief explanation"
    }}""")

_DETECTION_FOOTER_APPLICABILITY_NO_REASONING = dedent("""\
    You must evaluate the given attribute in two strict, sequential steps:

    Step 1: Check Object Existence (Applicability)
    - Decide whether the core object/context described in the attribute exists in the image. (e.g., If the attribute is about "how rocks are rendered", simply check if "rocks" exist).
    - If the object does NOT exist, the attribute is NOT applicable. Immediately set "applicable": false and "present": false. Do not look for the attribute itself.
    - If the object exists, set "applicable": true and proceed ONLY to Step 2.

    Step 2: Check Visual Presence (Only if Step 1 is true)
    - Only when "applicable" is true, evaluate whether the specific attribute is visually present on that object.
    - Focus purely on visual presence; do NOT judge overall quality or preference.

    Respond in the following JSON format:
    {{
        "applicable": <true|false>,
        "present": <true|false>,
        "confidence": <0.0 to 1.0>
    }}""")


def build_detection_prompt(
    attribute: str,
    prompt: str,
    use_prompt: bool = True,
    use_reasoning: bool = True,
    use_applicability: bool = False,
) -> str:
    """Build the detection user prompt.

    Args:
        attribute: Visual attribute to detect.
        prompt: Image generation prompt (included when use_prompt=True).
        use_prompt: Include the generation prompt context in the query.
        use_reasoning: Request a reasoning field in the JSON response.
        use_applicability: Ask the model to also report whether the image
            content allows the attribute to be evaluated at all (e.g. the
            attribute is about rocks but the image has no rocks). Downstream
            code maps applicable=false → -1 so callers can distinguish
            "doesn't have it" from "doesn't even apply".
    """
    header = _DETECTION_HEADER.format(attribute=attribute)
    prompt_block = _DETECTION_PROMPT_BLOCK.format(prompt=prompt) if use_prompt else ""
    if use_applicability:
        footer = (
            _DETECTION_FOOTER_APPLICABILITY_WITH_REASONING if use_reasoning
            else _DETECTION_FOOTER_APPLICABILITY_NO_REASONING
        )
    else:
        footer = (
            _DETECTION_FOOTER_WITH_REASONING if use_reasoning
            else _DETECTION_FOOTER_NO_REASONING
        )
    return header + prompt_block + footer


# Backward-compatible default (prompt included, reasoning included)
ATTRIBUTE_DETECTION_PROMPT = (
    _DETECTION_HEADER + _DETECTION_PROMPT_BLOCK + _DETECTION_FOOTER_WITH_REASONING
)
