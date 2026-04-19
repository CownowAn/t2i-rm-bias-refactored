import json
import datetime
import hashlib
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def edited_image_filename(attribute: str, prompt: str, baseline_path: str | Path) -> str:
    """Deterministic filename for an edited image based on attribute + prompt + baseline."""
    key = f"{attribute}|{prompt}|{Path(baseline_path).stem}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return f"edited_{h}.png"


def save_json(data: Any, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


def parse_json_response(resp: Any, marker: str = "json") -> tuple[Any, str | None]:
    """Extract JSON from LLM response. Returns (parsed, reasoning)."""
    import ast
    from json_repair import repair_json
    from loguru import logger

    raw_text = resp.first_response
    if raw_text is None:
        return None, None

    reasoning = resp.reasoning_content if hasattr(resp, "reasoning_content") else None

    try:
        if raw_text.strip().startswith(f"```{marker}"):
            json_str = raw_text.split(f"```{marker}", 1)[1].rsplit("```", 1)[0].strip()
            if reasoning is None:
                reasoning = raw_text.rsplit(f"```{marker}", 1)[0].strip()
        else:
            json_str = raw_text.strip()

        if marker == "python":
            try:
                output = ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                output = json.loads(repair_json(json_str))
        else:
            output = json.loads(repair_json(json_str))

    except Exception as e:
        output = raw_text.strip()
        if reasoning is None:
            reasoning = raw_text.strip()
        logger.warning(f"JSON parse error: {e}\nResponse: {raw_text[:200]}")

    return output, reasoning
