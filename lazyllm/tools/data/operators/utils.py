import regex

def boxed_res_extractor(text):
    if not isinstance(text, str):
        return None

    # Remove control characters: \x08oxed (\b==x08)
    text = regex.sub(r'[\p{C}]', '', text)

    # Accept different prefix: boxed{}, nboxed{}, etc
    pattern = (
        r'(?:\\?[a-z]*boxed|\\?[a-z]*oxed)'
        r'\{(?P<content>(?:[^{}]+|\{(?&content)\})*)\}'
    )

    matches = regex.findall(pattern, text, flags=regex.IGNORECASE)

    return matches[-1].strip() if matches else None
