import re
from typing import Any


_CHINESE_CHAR_LIMIT_RE = re.compile(
    r'(?P<prefix>不超过|至多|最多|约|大约|大概)?\s*'
    r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>万|千)?\s*字'
    r'(?P<suffix>左右|上下|以内|以下)?'
)
_NO_VISUALS = (
    re.compile(
        r'(?:不要|不需要|无需|不用|禁止)\s*(?:使用|添加|插入|生成|展示|显示)?\s*'
        r'(?:任何\s*)?(?:图片|图像|插图|配图|视觉(?:素材|内容)?)'
        r'|不(?:使用|添加|插入|生成|展示|显示)\s*(?:任何\s*)?'
        r'(?:图片|图像|插图|配图|视觉(?:素材|内容)?)|不插图|无图',
    ),
    re.compile(
        r'\b(?:no|without)\s+(?:any\s+)?(?:images?|pictures?|illustrations?|visuals?)\b'
        r"|\b(?:do\s+not|don't)\s+(?:use|include|add|generate|insert|show|display)\s+"
        r'(?:any\s+)?(?:images?|pictures?|illustrations?|visuals?)\b',
        re.IGNORECASE,
    ),
)
_REQUIRE_INPUT_IMAGE_REUSE = re.compile(
    r'(?:必须|务必|只能|仅限|只).{0,12}复用.{0,16}(?:我)?(?:上传(?:的)?(?:原图|图片|图像)|原图)'
    r'|(?:必须|务必|只能|仅限|只).{0,12}(?:使用|采用).{0,16}'
    r'(?:我)?(?:上传(?:的)?(?:原图|图片|图像)|原图).{0,20}(?:插入|放入|嵌入)'
    r'|(?:must|only).{0,20}reuse.{0,20}(?:uploaded|original).{0,12}(?:image|picture|photo)'
    r'|(?:must|only).{0,20}use.{0,20}(?:uploaded|original).{0,12}'
    r'(?:image|picture|photo).{0,20}(?:insert|embed|include)',
    re.IGNORECASE,
)
_FORBID_IMAGE_GENERATION = re.compile(
    r'(?:不要|禁止|不得).{0,12}(?:生成|改用|替换|替代).{0,12}(?:图|图片|图像)'
    r"|(?:do\s+not|don't|never).{0,20}(?:generate|replace|substitute).{0,20}"
    r'(?:image|picture|photo)',
    re.IGNORECASE,
)


def parse_writer_request_constraints(query: str) -> dict[str, Any]:
    """Parse stable Writer request constraints for every Writer entry point."""
    constraints: dict[str, Any] = {}
    match = _CHINESE_CHAR_LIMIT_RE.search(query or '')
    if match is not None:
        multiplier = {'万': 10000, '千': 1000}.get(match.group('unit'), 1)
        target_chars = int(float(match.group('value')) * multiplier)
        approximate = (
            match.group('prefix') in {'约', '大约', '大概'}
            or match.group('suffix') in {'左右', '上下'}
        )
        constraints.update({
            'target_chars': target_chars,
            'max_chars': target_chars * 11 // 10 if approximate else target_chars,
        })

    no_visuals = any(pattern.search(query or '') for pattern in _NO_VISUALS)
    require_reuse = bool(_REQUIRE_INPUT_IMAGE_REUSE.search(query or ''))
    forbid_generation = bool(_FORBID_IMAGE_GENERATION.search(query or ''))
    if no_visuals or require_reuse or forbid_generation:
        constraints['visual_policy'] = {
            'allow_visuals': not no_visuals,
            'require_input_image_reuse': require_reuse,
            'allow_image_generation': not (no_visuals or require_reuse or forbid_generation),
        }
    return constraints

