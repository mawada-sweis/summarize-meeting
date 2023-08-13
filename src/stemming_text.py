from nltk.stem import PorterStemmer

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Function to apply stemming to a list of words
def stemming_tokens(tokens_list: list) -> list:
    return [stemmer.stem(token) for token in tokens_list]
