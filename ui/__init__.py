from .interface import build_interface
from .events import register_events

#load_llm()

demo = build_interface(register_fn=register_events)