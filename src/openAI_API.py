import openai
import os
from src import remove_punctuation
from src import remove_stopwords
from src import tokanize_text
from src import var_operations
from src import tags
from src import stemming_text
from src import remove_filler_words

# Set your API key in .env file
if api_key := os.getenv("OPENAI_API_KEY"):
    openai.api_key = api_key
else:
    raise ValueError("OpenAI API key not found. Make sure the OPENAI_API_KEY environment variable is set.")

# Get all openai models
models_list = openai.Model.list()

# Keep only the ID of the models
models_ID = [model.id for model in models_list['data']]

def make_prompt_by_transcript(user_transcript: str, user_model_ID: str) -> str:
    """Generate a prompt for summarizing a meeting transcript.

    Args:
        user_transcript (str): The meeting transcript provided by the user.
        user_model_ID (str): The model ID provided by the user.

    Returns:
        str: A formatted prompt instructing the AI to summarize the meeting, including the
             transcript itself and guidelines for summarization.
    """
    # remove punctuation marks
    user_transcript = remove_punctuation.remove_from_str(
        text=user_transcript
    )
    
    # Tokenize the texts
    transcript_tokens = tokanize_text.tokenize(text=user_transcript,
                                             model_ID=user_model_ID
                                            )
    
    # Remove stopwords from texts
    clean_stopwords = remove_stopwords.remove_from_str_list(
        tokens=transcript_tokens
    )
    
    # Stemmed the transcript tokens
    stemmed_tokens = stemming_text.stemming_tokens(
        tokens_list=clean_stopwords
    )
    
    # Remove filler words from transcript tokens
    clean_filler = remove_filler_words.remove_from_list_str(
        tokens_list=stemmed_tokens
    )
    
    # Get the speakers tags
    transcript_tags = tags.get_tags_from_str(
        transcript=user_transcript
     )
    
    return f"""Please summarize the meeting transcript provided.
        Aim for a summary length of around 1 paragraphs.
        Write it in a bullet points
        
        here the transcript: 
        {user_transcript}

        speakers tags:
        {transcript_tags}

        Instructions:
        - Indicate speakers as participants tags.
        - Maintaining a natural and coherent writing style.
        - Use complete sentences and proper grammar.
        - Avoid redundant details or verbatim repetition of the transcript.
        
        Participants tags are:
        - pm is project manager,
        - ui is user interface designer,
        - me is marketing expert,
        - id is industrial designer.
        
        summary example:
            The project manager introduced the upcoming project to the team members
            and then the team members participated in an exercise in which they drew
            their favorite animal and discussed what they liked about the animal.
            The project manager talked about the project finances and selling prices.
            The team then discussed various features to consider in making the remote.
        """


def get_summarize_completion(transcript: str, model_id: str = "davinci"):
    """Generate a summarized version of a meeting transcript using an AI language model.

    Args:
        transcript (str): The meeting transcript to be summarized.
        model_id (str): The ID of the AI model to use for summarization. 
            Default is "davinci".

    Returns:
        str: The generated summarized text of the meeting, 
            based on the provided transcript and model.
    """
    if model_id not in models_ID:
        raise ValueError("Please choose a valid model ID.")

    transcript_tokens = tokanize_text.tokenize(text=transcript,
                            model_ID=model_id
    )
    max_token_value = tokanize_text.get_tokens_counts(transcript_tokens)
    
    prompt_value = make_prompt_by_transcript(
        user_transcript=transcript,
        user_model_ID=model_id
    )

    temperature_value = 0
    summarized_text = openai.Completion.create(
        model=model_id,
        prompt=prompt_value,
        temperature=temperature_value,
        max_tokens=600
        )
    
    return summarized_text["choices"][0]["text"], max_token_value
