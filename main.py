import gradio as gr
from src import openAI_API

MODELS_CHOICES = {
        'davinci',
        'text-davinci-001',
        'text-davinci-002', 
        'text-davinci-003'
}

demo = gr.Interface(
    fn=openAI_API.get_summarize_completion,
    inputs=[gr.Textbox(label="Transcript"),
            gr.Dropdown(
                    choices=MODELS_CHOICES,
                    value="davinci",
                    type="value",
                    label="Model ID"
                    )
        ],

    outputs="text",
    flagging_dir="./flagged/"
    )

if __name__ == "__main__":
    demo.launch()
