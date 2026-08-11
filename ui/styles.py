CUSTOM_CSS = """
#ingested_docs_table table td {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 200px;
    cursor: default;
}
/* unified input bar */

#chat_input_row {
    gap: 0 !important;
    align-items: stretch !important;
    border: 1.5px solid var(--border-color-primary, #444) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.18) !important;
    background: var(--input-background-fill, #1e1e2e) !important;
}

#chat_input_row #msg_input,
#chat_input_row #msg_input > label,
#chat_input_row #msg_input > label > div,
#chat_input_row #msg_input > div {
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
}

#chat_input_row #msg_input textarea,
#chat_input_row #msg_input input {
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 10px 14px !important;
    outline: none !important;
}
#chat_input_row #msg_input textarea:focus,
#chat_input_row #msg_input input:focus {
    box-shadow: none !important;
    outline: none !important;
}

#chat_input_row #send_btn,
#chat_input_row #stop_btn {
    border: none !important;
    border-left: 1px solid var(--border-color-primary, #444) !important;
    border-radius: 0 !important;
    min-height: 46px !important;
    align-self: stretch !important;
    min-width: unset !important;
    max-width: 90px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    background: var(--button-primary-background-fill, #2563eb) !important;
}
#chat_input_row #send_btn > span,
#chat_input_row #stop_btn > span {
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
}
#chat_input_row #send_btn:hover,
#chat_input_row #stop_btn:hover {
    filter: brightness(1.08) !important;
}
"""