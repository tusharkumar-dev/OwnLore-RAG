import threading
import gradio as gr

from vectordb.document_manager import list_all_pdfs
from config.prompt_config import _load_saved_prompt

def build_interface(register_fn=None):
    """
    Construct the full Gradio Blocks layout.

    Calls register_fn(demo, components) *inside* the gr.Blocks context so that
    Gradio 6+ event registration (demo.load, button.click, etc.) works correctly.
    Returns the fully wired demo object.
    """
    _basic_mode_prompt = _load_saved_prompt()

    with gr.Blocks(title="OwnLore-RAG") as demo:

        gr.Markdown("# OwnLore-RAG")

        with gr.Row():

            # ── left sidebar
            with gr.Column(scale=1, min_width=240):

                upload_button = gr.UploadButton(
                    "📎 Upload File(s)",
                    type="filepath",
                    file_count="multiple",
                    size="sm",
                )

                ingested_dataset = gr.Dataframe(
                    value=list_all_pdfs,
                    headers=["Document Name"],
                    label="Ingested Documents",
                    interactive=False,
                    max_height=250,
                    column_count=1,
                    elem_id="ingested_docs_table",
                )

                upload_notice = gr.Markdown(value="", elem_id="upload_notice")

                deselect_btn = gr.Button("📂 Chat with All Documents", size="sm")

                summarize_btn = gr.Button(
                    "📄 Summarize Selected Document",
                    variant="secondary", size="sm",
                )

                provider_dropdown = gr.Dropdown(
                    choices=[
                        ("Groq",        "groq"), 
                        ("Ollama",      "ollama"),
                        ("OpenAI",      "openai"),
                        ("Gemini",      "gemini"),
                        ("Anthropic",   "anthropic"),
                        ("OpenRouter",  "openrouter"),
                        ("DeepSeek",    "deepseek"),
                        ("Grok (xAI)",  "grok"),
                    ],
                    value="groq",
                    label="LLM Provider",
                )

                mode_toggle = gr.Radio(
                    choices=["🔵 RAG Mode", "🟢 Basic Mode", "🔍 Search Mode"],
                    value="🔵 RAG Mode",
                    label="Chat Mode",
                )

                with gr.Accordion("🔧 Advanced Settings", open=False):

                    top_k_slider = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Top K Chunks",
                        info="Controls Retrieved Chunks Panel (RAG + Search mode)",
                        visible=True,
                    )

                    show_sources_checkbox = gr.Checkbox(
                        label="📚 Show Sources & Chunks",
                        value=False,
                        visible=True,
                    )

                    advanced_prompt_input = gr.Textbox(
                        value=_basic_mode_prompt,
                        label="System Prompt",
                        lines=4,
                        visible=False,
                    )

                settings_btn = gr.Button("⚙️ RAG Settings", variant="secondary", size="sm")

                with gr.Accordion("🗂️ Document Management", open=False):

                    refresh_btn = gr.Button("🔄 Refresh List", size="sm")

                    with gr.Row():
                        delete_btn = gr.Button("🗑️ Delete Selected", variant="secondary", size="sm")

                    delete_all_btn = gr.Button("⚠️ Delete ALL", variant="stop")

                status = gr.Markdown(value="")

            # ── main chat column
            with gr.Column(scale=3):

                with gr.Row(elem_id="chat_header_row"):
                    with gr.Column(scale=9, min_width=0):
                        chat_context = gr.Markdown(value="## 💬 Chat\n📂 **All Documents**")
                    with gr.Column(scale=1, min_width=110):
                        clear_btn = gr.Button(
                            "🗑️ Clear Chat",
                            size="sm",
                            variant="secondary",
                            elem_id="clear_chat_btn",
                        )

                chatbot = gr.Chatbot(show_label=False, height=600)

                with gr.Row(elem_id="chat_input_row"):
                    msg_input = gr.Textbox(
                        placeholder="Ask anything...",
                        show_label=False,
                        scale=8,
                        lines=1,
                        elem_id="msg_input",
                    )
                    send_btn = gr.Button(
                        "Send ➤",
                        variant="primary",
                        scale=1,
                        visible=True,
                        elem_id="send_btn",
                    )
                    stop_btn = gr.Button(
                        "⏹ Stop",
                        variant="stop",
                        scale=1,
                        visible=False,
                        elem_id="stop_btn",
                    )

                gr.Markdown("_💡 Tip: Use RAG Mode for document-based questions._")

                # retrieval inspection panel — shows ALL Top-K chunks, not "sources used"
                with gr.Accordion("📚 Sources & Chunks", open=False, visible=False) as sources_accordion:
                    sources_panel = gr.Markdown(value="")

            # ── settings panel (hidden by default)
            with gr.Column(scale=1, visible=False, min_width=280) as settings_col:

                gr.Markdown("### ⚙️ RAG Settings")
                gr.Markdown("#### 📦 Chunk Settings")
                gr.Markdown("_⚠️ After changing, run Delete ALL and re-ingest._")

                chunk_size_input = gr.Slider(
                    minimum=200, maximum=2000, value=1000, step=100, label="Chunk Size",
                )
                chunk_overlap_input = gr.Slider(
                    minimum=0, maximum=500, value=200, step=50, label="Chunk Overlap",
                )

                gr.Markdown("#### 💬 Basic Mode System Prompt")
                prompt_input = gr.Textbox(value=_basic_mode_prompt, label="System Prompt", lines=5)

                gr.Markdown("#### 🧠 Embedding Model")

                embedding_input = gr.Dropdown(
                    choices=[
                        "BAAI/bge-small-en-v1.5",
                        "BAAI/bge-base-en-v1.5",
                        "BAAI/bge-large-en-v1.5",
                    ],
                    value="BAAI/bge-small-en-v1.5",
                    label="Embedding Model",
                )

                restore_btn = gr.Button("↩️ Restore Defaults", variant="secondary", size="sm")

                with gr.Row():
                    save_btn = gr.Button("💾 Save", variant="primary", size="sm")
                    cancel_btn = gr.Button("❌ Cancel", variant="secondary", size="sm")

                settings_status = gr.Textbox(label="", interactive=False, visible=False)

        # ── Gradio State objects
        chat_history      = gr.State([])
        pending_question  = gr.State("")
        stop_event        = gr.State(lambda: threading.Event())
        settings_open     = gr.State(False)
        saved_settings    = gr.State({
            "chunk_size":    1000,
            "chunk_overlap": 200,
            "prompt":        _basic_mode_prompt,
            "embedding":     "BAAI/bge-small-en-v1.5",
        })

        components = {
            # document sidebar
            "upload_button":         upload_button,
            "ingested_dataset":      ingested_dataset,
            "upload_notice":         upload_notice,
            "deselect_btn":          deselect_btn,
            "summarize_btn":         summarize_btn,
            "provider_dropdown":     provider_dropdown,
            "mode_toggle":           mode_toggle,
            "top_k_slider":          top_k_slider,
            "show_sources_checkbox": show_sources_checkbox,
            "advanced_prompt_input": advanced_prompt_input,
            "settings_btn":          settings_btn,
            "refresh_btn":           refresh_btn,
            "delete_btn":            delete_btn,
            "delete_all_btn":        delete_all_btn,
            "status":                status,
            # chat column
            "chat_context":          chat_context,
            "clear_btn":             clear_btn,
            "chatbot":               chatbot,
            "msg_input":             msg_input,
            "send_btn":              send_btn,
            "stop_btn":              stop_btn,
            "sources_accordion":     sources_accordion,
            "sources_panel":         sources_panel,
            # settings panel
            "settings_col":          settings_col,
            "chunk_size_input":      chunk_size_input,
            "chunk_overlap_input":   chunk_overlap_input,
            "prompt_input":          prompt_input,
            "embedding_input":       embedding_input,
            "restore_btn":           restore_btn,
            "save_btn":              save_btn,
            "cancel_btn":            cancel_btn,
            "settings_status":       settings_status,
            # state
            "chat_history":          chat_history,
            "pending_question":      pending_question,
            "stop_event":            stop_event,
            "settings_open":         settings_open,
            "saved_settings":        saved_settings,
        }

        if register_fn is not None:
            register_fn(demo, components)

    return demo