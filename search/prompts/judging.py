from textwrap import dedent

IMAGE_JUDGE_SYSTEM = (
    "You are an impartial judge with expertise in visual analysis. Your task is to compare two "
    "images and determine which one exhibits higher overall quality and is more preferable to "
    "human viewers."
)

IMAGE_JUDGE_PROMPT = dedent("""
    You are an impartial judge. Compare Image A and Image B and determine which is of higher
    overall quality and more preferable to human viewers, given the prompt they were generated from.

    The images were generated from this text prompt:
    <prompt>
    {prompt}
    </prompt>

    Judge purely on overall image quality and human preference — factors such as aesthetics,
    naturalness, composition, and how well the image matches the prompt. Do NOT judge based on any
    specific visual feature or editing style.

    Please provide your judgment in the following format:

    ```json
    {{
        "judgment": "<A|B|Tie>",
        "confidence": <0.0 to 1.0>,
        "reasoning": "Brief explanation of your judgment"
    }}
    ```

    Guidelines:
    - Return "A" if Image A is of higher quality and more preferable
    - Return "B" if Image B is of higher quality and more preferable
    - Return "Tie" if the quality difference is negligible

    Provide ONLY the JSON object, no additional explanations or markdown formatting.
""").strip()
