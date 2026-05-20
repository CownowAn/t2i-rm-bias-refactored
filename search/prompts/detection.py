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


def build_detection_prompt(
    attribute: str,
    prompt: str,
    use_prompt: bool = True,
    use_reasoning: bool = True,
) -> str:
    """Build the detection user prompt.

    Args:
        attribute: Visual attribute to detect.
        prompt: Image generation prompt (included when use_prompt=True).
        use_prompt: Include the generation prompt context in the query.
        use_reasoning: Request a reasoning field in the JSON response.
    """
    header = _DETECTION_HEADER.format(attribute=attribute)
    prompt_block = _DETECTION_PROMPT_BLOCK.format(prompt=prompt) if use_prompt else ""
    footer = _DETECTION_FOOTER_WITH_REASONING if use_reasoning else _DETECTION_FOOTER_NO_REASONING
    return header + prompt_block + footer


# Backward-compatible default (prompt included, reasoning included)
ATTRIBUTE_DETECTION_PROMPT = (
    _DETECTION_HEADER + _DETECTION_PROMPT_BLOCK + _DETECTION_FOOTER_WITH_REASONING
)
