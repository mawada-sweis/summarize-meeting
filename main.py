import gradio as gr
from src import openAI_API

MODELS_CHOICES = {
        'davinci',
        'text-davinci-001',
        'text-davinci-002', 
        'text-davinci-003',
        'text-davinci-edit-001',
        'gpt-3.5-turbo',
        'babbage', 
        'text-babbage-001', 
        'ada', 
        'text-ada-001', 
        'curie',
        'text-curie-001', 
}

FLAG_CHOICES = ['hallucination',
                'All points included',
                'Too brief', 
                'Too Long',
                'Redundunt',
                'Wrong Info',
                'Not complete'
                ]

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

    outputs=[gr.Textbox(label="Summary"),
             gr.Textbox(label="Tokens Count")
    ],
    flagging_dir="./flagged/",
    flagging_options=FLAG_CHOICES,
    )

if __name__ == "__main__":
    demo.launch()
