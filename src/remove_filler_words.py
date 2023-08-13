FILLER_WORDS = \
    ["um", "uh", "h", "mm", "like", "so", "well", "basically", "literally", "actually"]

def remove_from_list_str(tokens_list):
    return [token for token in tokens_list if token.lower() not in FILLER_WORDS]