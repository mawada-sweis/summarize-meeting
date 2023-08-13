import gradio as gr
from src import openAI_API

demo = gr.Interface(
    fn=openAI_API.get_summarize_completion,
    inputs=[gr.Textbox(label="Transcript"),
            gr.Dropdown(
                    choices={"gpt-3.5-turbo",
                             "gpt-3.5-turbo-16k",
                             "gpt-3.5-turbo-0613",
                             "gpt-3.5-turbo-16k-0613",
                             "text-davinci-003",
                             "text-davinci-002"
                             },
                    value="gpt-3.5-turbo",
                    type="value",
                    label="Model ID"
                    )
            ],

    outputs="text",
    flagging_dir="./flagged/"
    )

if __name__ == "__main__":
    demo.launch()
