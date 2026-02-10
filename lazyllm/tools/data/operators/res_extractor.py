import json5
import regex

def json_res_extractor(model_output,output_key):
    if isinstance(output_key, str):
        required_keys = [output_key]
    else:
        required_keys = [str(k) for k in output_key]
        if not required_keys:
            return {}

    text = model_output
    n = len(text)

    for start in range(n):
        if text[start] != "{":
            continue

        depth = 0
        in_string = False
        escape = False
        quote_char = ""

        for i in range(start, n):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
            else:
                if ch == '"' or ch == "'":
                    in_string = True
                    quote_char = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    if depth == 0:
                        break
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            obj = json5.loads(candidate)
                        except Exception:
                            break

                        if isinstance(obj, dict) and all(
                            k in obj for k in required_keys
                        ):
                            return obj
                        break
    return {}

def boxed_res_extractor(text):
    if not isinstance(text, str):
        return None
    pattern = r'\\boxed\{(?P<content>(?:[^{}]+|\{(?&content)\})*)\}'
    matches = regex.findall(pattern, text)
    return matches[-1].strip() if matches else None