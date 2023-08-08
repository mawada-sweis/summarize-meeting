import gradio as gr
import openai
import os


def check_model_id_existence(transcript: str = None, model_id: str = "gpt-4"):
    """Check if the chosen model ID is valid and exists 
    within the list of openai model IDs.

    Args:
        transcript (str): The meeting transcript to summarize.
        model_id (str): The selected model ID.

    Returns:
        str: A message indicating whether the model ID is valid or an error message if not.
    """
    return "Valid model choice" if model_id in models_ID else gr.Error("Please chose a valid model ID.")

# Set your API key in .env file
if api_key := os.getenv("OPENAI_API_KEY"):
    openai.api_key = api_key
else:
    raise ValueError("OpenAI API key not found. Make sure the OPENAI_API_KEY environment variable is set.")

# Get all openai models
models_list = openai.Model.list()

# Keep only the ID of the models
models_ID = [model.id for model in models_list['data']]

demo = gr.Interface(
    fn=check_model_id_existence, 
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
    flagging_dir="../flagged/"
    )

demo.launch()
