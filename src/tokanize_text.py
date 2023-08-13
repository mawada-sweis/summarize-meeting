import tiktoken


def tokenize(text: str, model_ID: str) -> list:
    """Tokenizes the input text into a list of tokens using 
    the provided model ID for encoding and decoding.

    Args:
        text (str): The input text to be tokenized.
        model_ID (str): The identifier of the model used for encoding.

    Returns:
        list: A list of tokens representing the tokenized input text.
    """
    enc = tiktoken.encoding_for_model(model_name=model_ID)
    tokens_encoded = tiktoken.get_encoding(enc.name).encode(text)
    return [enc.decode_single_token_bytes(token).decode('utf-8').strip() for token in tokens_encoded]


def get_max_tokens_counts(column: list) -> int:
    max_count = 0
    for tokens_list in column:
        count = len(tokens_list)
        if count > max_count:
            max_count = count
    return max_count