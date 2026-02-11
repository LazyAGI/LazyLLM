import regex

def boxed_res_extractor(text):
    if not isinstance(text, str):
        return None
    pattern = r'\\boxed\{(?P<content>(?:[^{}]+|\{(?&content)\})*)\}'
    matches = regex.findall(pattern, text)
    return matches[-1].strip() if matches else None