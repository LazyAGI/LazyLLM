from lazyllm import pipeline
from lazyllm.tools.data import pt


def build_structured_data_pipeline(llm, input_key='text', output_key='parsed', prompt=None):
    with pipeline() as ppl:
        ppl.text2json = pt.Text2Json(
            llm,
            input_key=input_key,
            output_key=output_key,
            prompt=prompt,
        )
    return ppl
