from textwrap import dedent

ATTRIBUTE_DETECTION_SYSTEM = (
    "You are an expert visual analyst. Your task is to determine whether a specific visual "
    "attribute is present in the provided image."
)

ATTRIBUTE_DETECTION_PROMPT = dedent("""
    Examine the provided image and determine whether it exhibits the following visual attribute.

    Attribute to detect: {attribute}

    The image was generated from this text prompt:
    <prompt>
    {prompt}
    </prompt>

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

    Respond with only the JSON block.
""").strip()
