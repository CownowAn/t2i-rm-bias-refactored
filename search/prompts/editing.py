from textwrap import dedent

EDIT_INSTRUCTION_SYSTEM = (
    "You are an expert in image editing and visual modification. Your task is to translate "
    "high-level visual feature descriptions into specific, actionable image editing instructions "
    "that can be applied to an image."
)

EDIT_INSTRUCTION_PROMPT = dedent("""
    You are given an image and a visual feature to apply to it.

    <feature>
    {attribute}
    </feature>

    The image was generated from this text prompt:
    <prompt>
    {prompt}
    </prompt>

    Write a single, concise instruction for an image editor that will modify the image to exhibit
    the feature above.

    Rules:
    - Target ONLY the given feature. Do NOT change the subject, composition, or any other aspect.
    - Use imperative form.
    - Be specific enough that an image editing model can follow it precisely.
    - Keep it concise.
    - The instruction must explicitly state that nothing other than the specified feature should change.

    Return ONLY the instruction string, with no additional explanation, quotes, or formatting.
""").strip()
