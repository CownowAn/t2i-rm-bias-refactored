from __future__ import annotations
import re
from pathlib import Path

from loguru import logger

from caller import AutoCaller, ChatHistory, ChatMessage
from search.prompts.editing import EDIT_INSTRUCTION_SYSTEM, EDIT_INSTRUCTION_PROMPT


class EditInstructionGenerator:
    """Uses a Vision LLM to turn a high-level attribute into an edit instruction."""

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        max_parallel: int = 1,
    ):
        self.model_name = model_name
        self.max_parallel = max_parallel
        self.caller = AutoCaller(dotenv_path=".env")

    async def generate(
        self,
        image_path: str,
        attribute: str,
        prompt_text: str,
    ) -> str:
        """Return a single-sentence instruction targeting only the given attribute."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        user_prompt = EDIT_INSTRUCTION_PROMPT.format(attribute=attribute, prompt=prompt_text)
        image_url = ChatMessage.image_to_base64_url(image_path)
        content = [
            {"type": "input_text", "text": user_prompt},
            {"type": "input_image", "image_url": image_url, "detail": "auto"},
        ]
        chat = ChatHistory(messages=[
            ChatMessage(role="system", content=EDIT_INSTRUCTION_SYSTEM),
            ChatMessage(role="user", content=content),
        ])

        responses = await self.caller.call(
            messages=[chat],
            model=self.model_name,
            max_parallel=self.max_parallel,
            max_tokens=512,
        )
        if not responses or responses[0] is None or not responses[0].has_response:
            logger.warning(f"Empty response for instruction generation; falling back to attribute text")
            return attribute

        instruction = responses[0].first_response.strip()
        instruction = re.sub(r'^["\']|["\']$', "", instruction).strip()
        logger.debug(f"Generated instruction: {instruction}")
        return instruction
