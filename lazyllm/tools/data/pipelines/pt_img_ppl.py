from lazyllm import pipeline
from lazyllm.tools.data import pt_mm


def build_mm_pt_pipeline(
        image_key='image_path',
        text_key='text',
        vlm=None,
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

    return ppl
