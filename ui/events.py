import gradio as gr

from vectordb.document_manager import list_all_pdfs
from config.prompt_config import _load_saved_prompt

from handlers.document_handler import (
    handle_ingest,
    handle_select,
    handle_deselect,
    handle_delete_selected,
    handle_delete_all,
)
from handlers.chat_handler import (
    respond,
    preflight,
    handle_stop,
    clear_chat,
    handle_summarize_to_chat,
    toggle_sources,
    handle_mode_change,
    ui_set_running,
    ui_set_idle,
)
from handlers.settings_handler import (
    handle_settings_save,
    toggle_settings_panel,
    _cancel_settings,
    sync_embedding_and_state,
)

# Restore-defaults constants — kept here because they only serve the restore_btn event
_DEFAULT_CHUNK_SIZE    = 1000
_DEFAULT_CHUNK_OVERLAP = 200


def _save_and_commit(chunk_size, chunk_overlap, prompt, embedding_model):
    """Wiring glue: call the settings handler then pack the new saved_settings dict."""
    col_update, status_update, status_bar_update = handle_settings_save(
        chunk_size, chunk_overlap, prompt, embedding_model
    )
    new_saved = {
        "chunk_size":    int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "prompt":        prompt,
        "embedding":     embedding_model,
    }
    return col_update, status_update, status_bar_update, new_saved


def _wire_send_chain(trigger, c):
    """Wire the four-step preflight → running → respond → idle chain onto `trigger`."""
    _respond_outputs = [c["chatbot"], c["sources_panel"], c["sources_accordion"], c["chat_history"]]
    _ui_outputs      = [c["send_btn"], c["stop_btn"], c["msg_input"], c["status"]]

    return (
        trigger(
            fn=preflight,
            inputs=[c["msg_input"], c["stop_event"]],
            outputs=[c["pending_question"]],
            queue=False,
            show_progress="hidden",
        )
        .then(
            fn=ui_set_running,
            inputs=None,
            outputs=_ui_outputs,
            queue=False,
            show_progress="hidden",
        )
        .then(
            fn=respond,
            inputs=[
                c["pending_question"], c["chat_history"],
                c["provider_dropdown"], c["mode_toggle"], c["top_k_slider"], c["stop_event"],
            ],
            outputs=_respond_outputs,
            show_progress="hidden",
        )
        .then(
            fn=ui_set_idle,
            inputs=None,
            outputs=_ui_outputs,
            queue=False,
            show_progress="hidden",
        )
    )


def register_events(demo, c):
    """
    Attach every Gradio event listener to the components in `c`.

    `demo`  — the gr.Blocks instance (needed for demo.load)
    `c`     — the components dict returned by build_interface()
    """
    _respond_outputs = [c["chatbot"], c["sources_panel"], c["sources_accordion"], c["chat_history"]]
    _ui_outputs      = [c["send_btn"], c["stop_btn"], c["msg_input"], c["status"]]

    # ── on load
    demo.load(
        fn=sync_embedding_and_state,
        outputs=[c["embedding_input"], c["saved_settings"]],
    )

    # ── document events
    c["upload_button"].upload(
        fn=handle_ingest,
        inputs=c["upload_button"],
        outputs=[c["ingested_dataset"], c["upload_notice"], c["status"]],
    )

    c["ingested_dataset"].select(fn=handle_select, outputs=c["chat_context"])
    c["deselect_btn"].click(fn=handle_deselect, outputs=c["chat_context"])
    c["refresh_btn"].click(fn=list_all_pdfs, outputs=c["ingested_dataset"])

    c["delete_btn"].click(
        fn=lambda: gr.update(value="🗑️ Deleting document..."),
        outputs=c["upload_notice"],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=handle_delete_selected,
        outputs=[c["ingested_dataset"], c["upload_notice"], c["status"], c["chat_context"]],
        show_progress="hidden",
    )

    c["delete_all_btn"].click(
        fn=lambda: gr.update(value="🗑️ Deleting all documents..."),
        outputs=c["upload_notice"],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=handle_delete_all,
        outputs=[c["ingested_dataset"], c["upload_notice"], c["status"], c["chat_context"]],
        show_progress="hidden",
    )

    # ── chat events
    c["show_sources_checkbox"].change(
        fn=toggle_sources,
        inputs=c["show_sources_checkbox"],
        outputs=[c["sources_panel"], c["sources_accordion"]],
        show_progress="hidden",
    )

    c["mode_toggle"].change(
        fn=handle_mode_change,
        inputs=c["mode_toggle"],
        outputs=[c["top_k_slider"], c["show_sources_checkbox"], c["advanced_prompt_input"]],
        show_progress="hidden",
    )

    c["advanced_prompt_input"].change(
        fn=lambda v: gr.update(value=v),
        inputs=c["advanced_prompt_input"],
        outputs=c["prompt_input"],
        show_progress="hidden",
    )

    c["clear_btn"].click(fn=clear_chat, outputs=[c["chatbot"], c["chat_history"]])

    c["summarize_btn"].click(
        fn=handle_summarize_to_chat,
        inputs=[c["chat_history"], c["provider_dropdown"]],
        outputs=[c["chatbot"], c["chat_history"]],
    )

    # ── settings events
    c["settings_btn"].click(
        fn=toggle_settings_panel,
        inputs=[c["settings_open"], c["saved_settings"]],
        outputs=[
            c["settings_col"], c["settings_open"],
            c["chunk_size_input"], c["chunk_overlap_input"],
            c["prompt_input"], c["embedding_input"],
        ],
        queue=False,
        show_progress="hidden",
    )

    c["cancel_btn"].click(
        fn=_cancel_settings,
        inputs=[c["saved_settings"]],
        outputs=[
            c["settings_col"], c["settings_open"],
            c["chunk_size_input"], c["chunk_overlap_input"],
            c["prompt_input"], c["embedding_input"],
        ],
        queue=False,
        show_progress="hidden",
    )

    # restore_btn — default prompt comes from disk so it matches initial state
    _default_prompt = _load_saved_prompt()
    c["restore_btn"].click(
        fn=lambda: (
            gr.update(value=_DEFAULT_CHUNK_SIZE),
            gr.update(value=_DEFAULT_CHUNK_OVERLAP),
            gr.update(value=_default_prompt),
        ),
        outputs=[c["chunk_size_input"], c["chunk_overlap_input"], c["prompt_input"]],
        queue=False,
        show_progress="hidden",
    )

    c["save_btn"].click(
        fn=_save_and_commit,
        inputs=[
            c["chunk_size_input"], c["chunk_overlap_input"],
            c["prompt_input"], c["embedding_input"],
        ],
        outputs=[c["settings_col"], c["settings_status"], c["status"], c["saved_settings"]],
    )
    c["save_btn"].click(
        fn=lambda: gr.update(visible=True),
        outputs=c["settings_status"],
    )
    c["save_btn"].click(
        fn=lambda: False,
        outputs=[c["settings_open"]],
        queue=False,
        show_progress="hidden",
    )

    # ── send chain
    _wire_send_chain(c["msg_input"].submit, c)
    _wire_send_chain(c["send_btn"].click, c)

    c["stop_btn"].click(
        fn=handle_stop,
        inputs=[c["stop_event"]],
        outputs=_ui_outputs,
        queue=False,
        show_progress="hidden",
    )