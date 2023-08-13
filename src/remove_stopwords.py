import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

def remove_from_str_list(tokens: list) -> list:
    """Remove English stopwords from a list of tokens.

    Args:
        tokens (list): A list of tokens.

    Returns:
        list: A new list of tokens with stopwords removed.
    """
    
    if 'en_stopwords' not in globals():
        en_stopwords = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in en_stopwords]
