from textwrap import dedent

EDIT_INSTRUCTION_SYSTEM = (
    "You are an expert image editor. Your task is to write a precise, minimal instruction "
    "that applies exactly one specified visual feature to an image, leaving everything else unchanged. "
    "Any modification beyond the specified feature is a failure."
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

    *** CRITICAL CONSTRAINT: Apply ONLY the feature above. ALL other visual elements must remain exactly as they are.
    Any change beyond the specified feature is a failure. ***

    Write a single, concise instruction for an image editor that will modify the image to exhibit
    the feature above.

    Rules:
    - Target ONLY the given feature; preserve everything else.
    - Use imperative form.
    - Be specific enough that an image editing model can follow it precisely.
    - Keep it concise.
    - The instruction MUST explicitly state that nothing other than the specified feature should change.

    Return ONLY the instruction string, with no additional explanation, quotes, or formatting.
""").strip()
