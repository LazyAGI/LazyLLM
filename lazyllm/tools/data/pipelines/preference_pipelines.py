from lazyllm import pipeline
from lazyllm.tools.data import preference_ops


def build_preference_pipeline(model, input_key='content', n=3, temperature=1.0,
                              strategy='max_min', threshold=0.5):
    with pipeline() as ppl:
        ppl.intent_extractor = preference_ops.IntentExtractor(model=model, input_key=input_key)
        ppl.preference_response_generator = preference_ops.PreferenceResponseGenerator(
            model=model,
            n=n,
            temperature=temperature
        )
        ppl.response_evaluator = preference_ops.ResponseEvaluator(model=model)
        ppl.preference_pair_constructor = preference_ops.PreferencePairConstructor(
            strategy=strategy,
            threshold=threshold
        )
    return ppl
