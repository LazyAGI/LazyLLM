VISION_SUMMARY_PROMPT = (
    'Describe the visible content of this image for downstream document writing. '
    'Include visible text, important objects, layout, charts, and relationships. '
    'Do not infer facts that are not visible. Return plain text only.'
)


RESOLVE_VISUAL_NEEDS_PROMPT = '''Match available media to visual needs.

Return an object whose selections maps each need_id to zero or more media_asset_id values.
Use only IDs in the media catalog. Select an asset only when its summary clearly satisfies the need.
Do not invent IDs, paths, URLs, or new visual needs.

Visual needs:
{visual_needs_json}

Available media:
{available_media_json}
'''
