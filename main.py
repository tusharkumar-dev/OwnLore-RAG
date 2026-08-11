import gradio as gr
from ui import demo
from ui.styles import CUSTOM_CSS

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        debug=False,
        show_error=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )