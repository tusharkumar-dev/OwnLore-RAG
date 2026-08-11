from typing import Generator
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm.llm_provider import load_llm, get_chunk_text
from config.state import get_basic_prompt

_SYSTEM_PROMPT = """
You are a helpful, knowledgeable, and friendly assistant.
Answer the user's questions clearly and concisely.
You can use your general knowledge to answer.
If you don't know something, say so honestly.
"""

def basic_chat(question: str, history, provider: str = None) -> Generator[str, None, None]:

    prompt = get_basic_prompt() or _SYSTEM_PROMPT

    if not question.strip():
        yield "Koi question nahi diya."
        return

    llm = load_llm(provider=provider)

    msgs = [SystemMessage(content=prompt)]

    for human, asst in history:
        if human:
            msgs.append(HumanMessage(content=human))
        if asst:
            msgs.append(AIMessage(content=asst))

    msgs.append(HumanMessage(content=question))

    # start streaming now
    acc = ""
    for chunk in llm.stream(msgs):
        acc += get_chunk_text(chunk)
        yield acc
