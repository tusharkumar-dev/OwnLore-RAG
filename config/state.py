_basic_mode_prompt = """You are a helpful, knowledgeable, and friendly assistant.
Answer the user's questions clearly and concisely."""

def get_basic_prompt() -> str:
    return _basic_mode_prompt

def set_basic_prompt(prompt: str) -> None:
    global _basic_mode_prompt
    _basic_mode_prompt = prompt.strip() or _basic_mode_prompt