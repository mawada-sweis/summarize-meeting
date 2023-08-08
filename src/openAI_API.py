import openai
import os

# Set your API key in .env file
if api_key := os.getenv("OPENAI_API_KEY"):
    openai.api_key = api_key
else:
    raise ValueError("OpenAI API key not found. Make sure the OPENAI_API_KEY environment variable is set.")

# Get all openai models
models_list = openai.Model.list()

# Keep only the ID of the models
models_ID = [model.id for model in models_list['data']]


def get_summarize_completion(transcript: str = None, model_id: str = "gpt-3.5-turbo"):
    """

    Args:
        transcript (str): The meeting transcript to summarize.
        model_id (str): The selected model ID.

    Returns:
        str: 
    """
    if model_id in models_ID:
        temperature_value = 1
        max_token_value = 200
        prompt_value = f"Summarize following text which is a meeting transcript: {transcript}"
        summarized_text = openai.Completion.create(
            model=model_id,
            prompt=prompt_value,
            temperature=temperature_value,
            max_tokens = max_token_value
        )
        return summarized_text["choices"][0]["text"]
    else:
        gr.Error("Please chose a valid model ID.")
