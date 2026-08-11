import json
from pathlib import Path

_PROMPT_CONFIG_PATH = Path(__file__).parent / "prompt_config.json"

_DEFAULT_BASIC_PROMPT = """You are a helpful, knowledgeable, and friendly assistant.
Answer the user's questions clearly and concisely."""


def _load_saved_prompt() -> str:
    try:
        if _PROMPT_CONFIG_PATH.exists():
            data = json.loads(_PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
            return data.get("basic_mode_prompt", _DEFAULT_BASIC_PROMPT)
    except Exception:
        pass
    return _DEFAULT_BASIC_PROMPT


def _save_prompt_to_disk(prompt: str) -> None:
    try:
        _PROMPT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PROMPT_CONFIG_PATH.write_text(
            json.dumps({"basic_mode_prompt": prompt}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass