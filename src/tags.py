import re


def get_tags(column: list) -> list:
    tags = list(set(column.str.findall(r'<(.*?)>').sum()))
    return [tag.lower() for tag in tags]


def get_tags_from_str(transcript: str) -> list:
    return list({tag.lower() for tag in re.findall(r'<(.*?)>', transcript)})
