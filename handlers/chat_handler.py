import threading
import gradio as gr

from chat.rag_chat import answer_question
from chat.basic_chat import basic_chat
from chat.search_mode import search_chunks
from chat.summarize import summarize_pdf
from handlers.document_handler import get_selected_pdf

# module-level state
_show_sources = False
_last_sources = ""

# ── message helpers

def get_role(msg):
    if isinstance(msg, dict):
        return msg.get("role", "")
    return getattr(msg, "role", "")


def get_content(msg):
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


def make_msg(role, content):
    return {"role": role, "content": content}


def _build_history(history):
    history_tuples = []
    clean = history[:-2]
    i = 0
    while i < len(clean) - 1:
        u, a = clean[i], clean[i + 1]
        if get_role(u) == "user" and get_role(a) == "assistant":
            history_tuples.append((get_content(u), get_content(a)))
            i += 2
        else:
            i += 1
    return history_tuples

# ── sources helpers

def _format_retrieved_chunks(src_text: str) -> str:
    # rag_chat.py already formats source_formatter output correctly.
    # passing through unchanged — don't touch this
    if not src_text or not src_text.strip():
        return "_Ask a question first to see retrieved chunks..._"
    return src_text

def toggle_sources(checked: bool):

    global _show_sources
    _show_sources = checked
    if checked and _last_sources:
        return (
            gr.update(visible=True, value=_format_retrieved_chunks(_last_sources)),
            gr.update(visible=True, open=True),
        )
    elif checked:
        return (
            gr.update(visible=True, value="_Ask a question first to see retrieved chunks..._"),
            gr.update(visible=True, open=False),
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
        )

# ── mode

def handle_mode_change(mode):
    retrieval = mode in ("🔵 RAG Mode", "🔍 Search Mode")
    return (
        gr.update(visible=retrieval),      # top_k_slider
        gr.update(visible=retrieval),      # show_sources_checkbox
        gr.update(visible=not retrieval),  # advanced_prompt_input
    )


# ── summarize

def handle_summarize_to_chat(history, provider):
    _selected_pdf = get_selected_pdf()
    if not _selected_pdf:
        err_history = list(history)
        err_history.append(make_msg("user", "📄 Summarize Selected Document"))
        err_history.append(make_msg("assistant", "❌ Please select a document first."))
        yield err_history, err_history
        return

    history = list(history)
    history.append(make_msg("user", "📄 Summarize Selected Document"))
    history.append(make_msg("assistant", ""))

    stream_window = [history[-2], history[-1]]
    yield stream_window, gr.update()

    accumulated = ""
    for chunk in summarize_pdf(doc_name=_selected_pdf, provider=provider):
        accumulated = chunk
        history[-1] = make_msg("assistant", accumulated)
        stream_window[-1] = history[-1]
        yield stream_window, gr.update()

    yield history, history


# ── clear

def clear_chat():
    return [], []


# ── send / stop / preflight

def preflight(question, stop_event: threading.Event):
    stop_event.clear()
    return question.strip()

def handle_stop(stop_event: threading.Event):
    stop_event.set()
    return _ui_set_idle()

def _ui_set_running():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value="", interactive=False),
        gr.update(value="🟡 Generating Response..."),
    )

def _ui_set_idle():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(interactive=True),
        gr.update(value=""),
    )

def respond(question, history, provider, mode, top_k, stop_event: threading.Event):
    
    global _last_sources

    _selected_pdf = get_selected_pdf()

    no_commit = gr.update()
    sources_update = gr.update()
    accordion_update = gr.update()

    if not question:
        yield history, gr.update(), gr.update(), history
        return

    history = list(history)
    history.append(make_msg("user", question))
    history.append(make_msg("assistant", ""))

    stream_window = [history[-2], history[-1]]
    yield stream_window, sources_update, accordion_update, no_commit

    if mode == "🔍 Search Mode":
        result = search_chunks(question, selected_pdf=_selected_pdf, top_k=int(top_k))
        history[-1] = make_msg("assistant", result)
        yield history, gr.update(visible=False), gr.update(visible=False), history
        return

    if mode == "🟢 Basic Mode":
        history_tuples = _build_history(history)
        try:
            for answer in basic_chat(question, history_tuples, provider=provider):
                if stop_event.is_set():
                    break
                history[-1] = make_msg("assistant", answer)
                stream_window[-1] = history[-1]
                yield stream_window, sources_update, accordion_update, no_commit
        except ValueError as e:
            # missing API key for the selected provider
            history[-1] = make_msg("assistant", f"⚠️ {e}")
            yield history, gr.update(), gr.update(), history
            return
        except Exception as e:
            history[-1] = make_msg("assistant", f"❌ Something went wrong: {e}")
            yield history, gr.update(), gr.update(), history
            return
        yield history, sources_update, accordion_update, history
        return

    try:
        for answer, sources in answer_question(
            question, [],
            selected_pdf=_selected_pdf,
            provider=provider,
            top_k=int(top_k),
        ):
            if stop_event.is_set():
                break

            _last_sources = sources
            history[-1] = make_msg("assistant", answer)
            stream_window[-1] = history[-1]

            if _show_sources:
                sources_update = gr.update(value=_format_retrieved_chunks(sources), visible=True)
                accordion_update = gr.update(visible=True, open=True)

            yield stream_window, sources_update, accordion_update, no_commit
    except ValueError as e:
        # missing API key for the selected provider — show a clean message
        history[-1] = make_msg("assistant", f"⚠️ {e}")
        yield history, gr.update(), gr.update(), history
        return
    except Exception as e:
        history[-1] = make_msg("assistant", f"❌ Something went wrong: {e}")
        yield history, gr.update(), gr.update(), history
        return

    yield history, sources_update, accordion_update, history


# expose ui helpers
ui_set_running = _ui_set_running
ui_set_idle = _ui_set_idle