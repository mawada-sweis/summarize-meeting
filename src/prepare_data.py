from datasets import load_dataset
import pandas as pd
import tiktoken
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')

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


def remove_stop_words(tokens: list) -> list:
    """Remove English stopwords from a list of tokens.

    Args:
        tokens (list): A list of tokens.

    Returns:
        list: A new list of tokens with stopwords removed.
    """
    
    if 'en_stopwords' not in globals():
        en_stopwords = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in en_stopwords]


dataset = load_dataset('TalTechNLP/AMIsum', split='train+validation+test').to_pandas()
dataset = pd.DataFrame(dataset.drop(columns='id')) 

dataset['summary_tokens'] = dataset['summary'].apply(tokenize_text, model_ID=model_id)
dataset['transcript_tokens'] = dataset['transcript'].apply(tokenize_text, model_ID=model_id)

# Get the talks position tag
tags = set(dataset['transcript'].str.findall(r'<(.*?)>').sum())
tags = [f'<{tag}>' for tag in tags]

dataset['clean_transcript'] = dataset['transcript_tokens'].apply(remove_stop_words)
dataset.to_csv('./dataset/cleaned_dataset.csv', index=False)