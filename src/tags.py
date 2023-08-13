def get_tags(column: list) -> list:
    tags = set(column.str.findall(r'<(.*?)>').sum())
    return [f'<{tag}>' for tag in tags]