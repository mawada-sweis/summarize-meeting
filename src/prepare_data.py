from datasets import load_dataset
import pandas as pd
import tiktoken

# Set the model ID to a fixed value for this scenario
model_id = 'davinci'


def tokenize_text(text: str, model_ID: str) -> list:
    """Tokenizes the input text into a list of tokens using 
    the provided model ID for encoding and decoding.

    Args:
        text (str): The input text to be tokenized.
        model_ID (str): The identifier of the model used for encoding.

    Returns:
        list: A list of tokens representing the tokenized input text.
    """
    enc = tiktoken.encoding_for_model(model_name=model_ID)
    tokens = tiktoken.get_encoding(enc.name).encode(text)
    tokens = [enc.decode_single_token_bytes(token).decode('utf-8').strip() for token in tokens]
    return tokens

dataset = load_dataset('TalTechNLP/AMIsum', split='train+validation+test').to_pandas()
dataset = pd.DataFrame(dataset.drop(columns='id')) 

dataset['summary_tokens'] = dataset['summary'].apply(tokenize_text, model_ID=model_id)
dataset['transcript_tokens'] = dataset['transcript'].apply(tokenize_text, model_ID=model_id)

# Get the talks position tag
tags = set(transcript_dataset['transcript'].str.findall(r'<(.*?)>').sum())
tags = [f'<{tag}>' for tag in tags]
