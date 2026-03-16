from lazyllm import pipeline
from lazyllm.tools.data import pt, pt_mm


def build_mm_pt_pipeline(
        context_key='context',
        image_key='image_path',
        text_key='text',
        vlm=None,
        num_qa=5,
        min_width=256,
        min_height=256,
        max_side=1024,
        relevance_threshold=0.6,
        use_dedup=True):
    with pipeline() as ppl:
        ppl.integrity_check = pt_mm.integrity_check(
            image_key=image_key
        )

        ppl.resolution_filter = pt_mm.resolution_filter(
            image_key=image_key,
            min_width=min_width,
            min_height=min_height
        )

        ppl.resolution_resize = pt_mm.resolution_resize(
            image_key=image_key,
            max_side=max_side
        )

        if use_dedup:
            ppl.image_dedup = pt_mm.ImageDedup(image_key=image_key)

        if vlm is not None:
            ppl.text_relevance_filter = pt_mm.TextRelevanceFilter(
                vlm=vlm,
                image_key=image_key,
                text_key=text_key,
                threshold=relevance_threshold
            )

            ppl.context_qual_filter = pt.ContextQualFilter(
                llm=vlm,
                context_key=context_key,
                image_key=image_key
            )

            ppl.phi4_qa_generator = pt.Phi4QAGenerator(
                llm=vlm,
                context_key=context_key,
                image_key=image_key,
                num_qa=num_qa
            )

    return ppl
