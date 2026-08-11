from dotenv import load_dotenv
import os
from config.settings import (
    LLM_PROVIDER, LLM_TEMPERATURE, LLM_STREAMING, _MODEL_MAP
)

#api key map
_API_KEY_MAP = {
    "groq":       ("GROQ_API_KEY", "https://console.groq.com"),
    "openai":     ("OPENAI_API_KEY", "https://platform.openai.com"),
    "anthropic":  ("ANTHROPIC_API_KEY", "https://console.anthropic.com"),
    "gemini":     ("GOOGLE_API_KEY", "https://aistudio.google.com"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai"),
    "deepseek":   ("DEEPSEEK_API_KEY", "https://platform.deepseek.com"),
    "grok":       ("XAI_API_KEY", "https://console.x.ai"),
}

#api key missing
def _validate_api_key(provider: str) -> str | None:

    if provider not in _API_KEY_MAP:
        return None

    env_var, url = _API_KEY_MAP[provider]
    if not os.environ.get(env_var):
        return (
            f"API key missing for provider '{provider}'. "
            f"Set {env_var} in your .env file. Get your key: {url}"
        )
    return None


def load_llm(provider: str = None):

    load_dotenv(override=True)

    active_provider = provider or LLM_PROVIDER
    model = _MODEL_MAP.get(active_provider, "llama3")

    key_error = _validate_api_key(active_provider)
    if key_error:
        raise ValueError(key_error)

    if active_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=LLM_TEMPERATURE,
            model=model,
            streaming=LLM_STREAMING,
        )

    elif active_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=LLM_TEMPERATURE,
        )

    elif active_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=LLM_TEMPERATURE,
            streaming=LLM_STREAMING,
        )

    elif active_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=LLM_TEMPERATURE,
            streaming=LLM_STREAMING,
    )

    elif active_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=LLM_TEMPERATURE,
            streaming=LLM_STREAMING,
        )

    elif active_provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=LLM_TEMPERATURE,
            streaming=LLM_STREAMING,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

    elif active_provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=LLM_TEMPERATURE,
            streaming=LLM_STREAMING,
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
        )

    elif active_provider == "grok":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=LLM_TEMPERATURE,
            streaming=LLM_STREAMING,
            base_url="https://api.x.ai/v1",              # ← xAI endpoint
            api_key=os.environ.get("XAI_API_KEY"),
        )

    else:
        raise ValueError(f"Unknown provider: '{active_provider}'")

def get_chunk_text(chunk) -> str:
   
    content = getattr(chunk, "content", chunk)
 
    if isinstance(content, str):
        return content
 
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
 
    return str(content) if content is not None else ""