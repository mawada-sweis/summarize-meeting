import string
import re

def remove_from_str(text: str) -> str:
    cleaned_text = ''.join([char for char in text if char not in string.punctuation])
    return re.sub(r'\s+', ' ', cleaned_text).strip()