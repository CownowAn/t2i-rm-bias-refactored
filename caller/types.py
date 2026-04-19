"""
Types for LLM API calls.

Reference:
https://openrouter.ai/docs/api-reference/overview

TODO:
* Add streaming support
"""

import json
import base64
from pathlib import Path
from loguru import logger
from typing import Sequence, Any, Literal, Optional, Union
from pydantic import BaseModel


class FunctionDescription(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Any


class Tool(BaseModel):
    type: Literal["function"]
    function: FunctionDescription


class FunctionName(BaseModel):
    name: str


class ToolChoiceFunction(BaseModel):
    type: Literal["function"]
    function: FunctionName


ToolChoice = Union[Literal["none"], Literal["auto"], ToolChoiceFunction]


class ResponseFormat(BaseModel):
    type: Literal["json_schema"]
    json_schema: dict


class InferenceConfig(BaseModel):
    """
    All the optional parameters.
    """

    response_format: Optional[ResponseFormat] = None
    stop: Optional[list] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    tools: Optional[list[Tool]] = None
    tool_choice: Optional[ToolChoice] = None

    # Sampling parameters
    seed: Optional[int] = None
    top_p: Optional[float] = None  # (0, 1]
    frequency_penalty: Optional[float] = None  # [-2, 2]
    presence_penalty: Optional[float] = None  # [-2, 2]
    repetition_penalty: Optional[float] = None  # (0, 2]
    min_p: Optional[float] = None  # [0, 1]
    top_a: Optional[float] = None  # [0, 1]
    logit_bias: Optional[dict[int, float]] = None
    top_logprobs: Optional[int] = None

    # Extra body
    reasoning: Optional[str | int] = None
    extra_body: Optional[dict] = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, list]  # str (텍스트) 또는 list (multimodal)

    @staticmethod
    def image_to_base64_url(image_path: str) -> str:
        """Convert image file to base64 data URL for vision models."""
        image_path = Path(image_path)
        suffix = image_path.suffix.lower()

        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{b64}"

    def as_text(self) -> str:
        if isinstance(self.content, str):
            return f"{self.role}:\n{self.content}"
        else:
            text_parts = []
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "input_text":
                    text_parts.append(item.get("text", ""))
            return f"{self.role}:\n" + " ".join(text_parts)

    def to_openai_content(self) -> dict:
        if isinstance(self.content, str):
            return {
                "role": self.role,
                "content": self.content,
            }
        else:
            return {
                "role": self.role,
                "content": self.content,
            }


class ToolMessage(BaseModel):
    role: Literal["tool"]
    content: str
    tool_call_id: str
    name: Optional[str] = None

    def as_text(self) -> str:
        return f"{self.role}:\n{self.content}"

    def to_openai_content(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }


Message = Union[ChatMessage, ToolMessage]


class ChatHistory(BaseModel):
    messages: Sequence[Message] = []

    def as_text(self) -> str:
        return "\n".join([msg.as_text() for msg in self.messages])

    @staticmethod
    def from_system(content: str) -> "ChatHistory":
        return ChatHistory(messages=[ChatMessage(role="system", content=content)])

    @staticmethod
    def from_user(content: str) -> "ChatHistory":
        return ChatHistory(messages=[ChatMessage(role="user", content=content)])

    @staticmethod
    def from_user_with_images(text: str, image_paths: list[str]) -> "ChatHistory":
        """
        Create a user message with images.

        Args:
            text: User message text
            image_paths: List of image file paths to include

        Returns:
            ChatHistory with multimodal content
        """
        content = [
            {
                "type": "input_text",
                "text": text,
            }
        ]

        # Add images in base64 format
        for image_path in image_paths:
            try:
                image_url = ChatMessage.image_to_base64_url(image_path)
                content.append({
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": "auto",
                })
            except Exception as e:
                logger.warning(f"Failed to convert image {image_path} to base64: {e}")

        return ChatHistory(messages=[ChatMessage(role="user", content=content)])

    def remove_system(self) -> "ChatHistory":
        """Remove all system prompts and creates a new copy."""
        new_messages = []
        for msg in self.messages:
            if msg.role != "system":
                new_messages.append(msg.model_copy())
        assert not any(msg.role == "system" for msg in new_messages)

        return ChatHistory(messages=new_messages)

    def add_user(self, content: str) -> "ChatHistory":
        new_messages = list(self.messages) + [ChatMessage(role="user", content=content)]
        return ChatHistory(messages=new_messages)

    def add_assistant(self, content: str) -> "ChatHistory":
        new_messages = list(self.messages) + [ChatMessage(role="assistant", content=content)]
        return ChatHistory(messages=new_messages)

    def add_messages(self, messages: Sequence[ChatMessage]) -> "ChatHistory":
        new_messages = list(self.messages) + list(messages)
        return ChatHistory(messages=new_messages)

    def to_openai_messages(self) -> list[dict]:
        return [msg.to_openai_content() for msg in self.messages]
    
    def to_openai_str(self) -> str:
        return json.dumps(self.to_openai_messages(), indent=4)

    def get_first(self, role: Literal["system", "user", "assistant"]) -> str | None:
        """
        Get the first message with the given role, if exists.
        Returns None otherwise.
        For multimodal messages, extracts text content only.
        """
        for msg in self.messages:
            if msg.role == role:
                if isinstance(msg.content, str):
                    return msg.content
                else:
                    # Multimodal 경우, 텍스트만 추출
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "input_text":
                            return item.get("text")
        return None


class Request(BaseModel):
    """
    Main request format for OpenRouter.
    """

    model: str
    messages: list[Message] | ChatHistory
    config: InferenceConfig

    def to_openrouter_request(self) -> dict:
        request_body = {"model": self.model}
        if isinstance(self.messages, ChatHistory):
            request_body["messages"] = self.messages.to_openai_messages()
        else:
            request_body["messages"] = [msg.to_openai_content() for msg in self.messages]

        config_dict = self.config.model_dump()

        if config_dict["reasoning"] is None:
            pass
        elif isinstance(config_dict["reasoning"], int):
            if config_dict["extra_body"] is None:
                config_dict["extra_body"] = {}
            config_dict["extra_body"]["reasoning"] = {"max_tokens": config_dict["reasoning"]}
        elif isinstance(config_dict["reasoning"], str):
            if config_dict["extra_body"] is None:
                config_dict["extra_body"] = {}
            config_dict["extra_body"]["reasoning"] = {"effort": config_dict["reasoning"]}

        config_dict.pop("reasoning")

        request_body.update(config_dict)
        return request_body

    def to_openai_request(self) -> dict:
        request_body = {"model": self.model}
        if self.model.startswith("openai/"):
            print("Please remove the 'openai/' prefix from the model name when using OpenAICaller.")
            self.model = self.model.removeprefix("openai/")

        if isinstance(self.messages, ChatHistory):
            request_body["input"] = self.messages.to_openai_messages()
        else:
            request_body["input"] = [msg.to_openai_content() for msg in self.messages]

        config_dict = self.config.model_dump()
        config_dict["max_output_tokens"] = config_dict.pop("max_tokens")

        if config_dict["reasoning"] is None:
            pass
        elif isinstance(config_dict["reasoning"], int):
            logger.warning("Reasoning should be a string, not an integer, for OpenAICaller. Using 'medium' instead.")
            config_dict["reasoning"] = {"effort": "medium", "summary": "auto"}
        elif isinstance(config_dict["reasoning"], str):
            config_dict["reasoning"] = {"effort": config_dict["reasoning"], "summary": "auto"}

        request_body.update(config_dict)
        return request_body

    def to_anthropic_request(self) -> dict:
        request_body = {"model": self.model}
        if self.model.startswith("anthropic/"):
            print("Please remove the 'anthropic/' prefix from the model name when using AnthropicCaller.")
            self.model = self.model.removeprefix("anthropic/")

        if isinstance(self.messages, ChatHistory):
            request_body["messages"] = self.messages.to_openai_messages()
        else:
            request_body["messages"] = [msg.to_openai_content() for msg in self.messages]

        config_dict = self.config.model_dump()

        if config_dict["reasoning"] is None:
            config_dict.pop("reasoning")
            pass
        elif isinstance(config_dict["reasoning"], int):
            assert config_dict["max_tokens"] >= 1024, "Max tokens must be at least 1024 for reasoning"
            config_dict["thinking"] = {"budget_tokens": config_dict.pop("reasoning"), "type": "enabled"}
        elif isinstance(config_dict["reasoning"], str):
            config_dict.pop("reasoning")
            logger.warning("Reasoning should be an integer, not a string, for AnthropicCaller. Defaulting to 1024 thinking tokens instead.")
            config_dict["thinking"] = {"budget_tokens": 1024, "type": "enabled"}

        request_body.update(config_dict)
        return request_body   

class NonStreamingChoice(BaseModel):
    model_config = {"extra": "allow"}

    finish_reason: Optional[str] = None
    native_finish_reason: Optional[str] = None
    message: dict[str, Any]
    error: Optional[dict] = None


class Response(BaseModel):
    """Unified response format for all providers."""

    model_config = {"extra": "allow"}

    id: str
    choices: list[NonStreamingChoice]
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    usage: dict

    @property
    def first_choice(self) -> NonStreamingChoice | None:
        """Returns the first choice of the response."""
        if len(self.choices) == 0:
            logger.warning(f"No choices found in Response {self.id} from {self.model}")
            return None
        return self.choices[0]

    @property
    def first_response(self) -> str | None:
        """Returns the first response's content if it exists, otherwise None."""
        first_choice = self.first_choice
        if first_choice is None:
            return None
        content = first_choice.message["content"]
        if content is None:
            logger.warning(f"No content found in first choice of Response {self.id} from {self.model}: {first_choice}")
        return content

    @property
    def has_response(self) -> bool:
        return self.first_response is not None

    @property
    def reasoning_content(self) -> str | None:
        """Returns the first response's reasoning content if it exists, otherwise None."""
        first_choice = self.first_choice
        if first_choice is None:
            return None
        if "reasoning_details" not in first_choice.message:
            logger.info(f"No reasoning details found in first choice of Response: {self}")
            return None

        reasoning_details = first_choice.message["reasoning_details"][0]
        if reasoning_details["type"] == "reasoning.summary":
            return reasoning_details["summary"]
        elif reasoning_details["type"] == "reasoning.text":
            return reasoning_details["text"]
        elif reasoning_details["type"] == "reasoning.encrypted":
            logger.debug(f"Reasoning details are encrypted in Response: {self}")
            return reasoning_details["data"]

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning_content is not None

    @property
    def finish_reason(self) -> str | None:
        """
        Returns the finish reason of the response.

        Possible values: "stop", "length", "content_filter", "error", "tool_calls".
        """
        first_choice = self.first_choice
        if first_choice is None:
            return None
        finish_reason = first_choice.finish_reason or first_choice.native_finish_reason
        return finish_reason
    
    def __str__(self) -> str:
        return self.first_response or ""
