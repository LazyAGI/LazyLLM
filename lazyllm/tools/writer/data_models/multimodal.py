from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from ..utils.artifact import ArtifactModel


class MediaAsset(BaseModel):
    media_asset_id: str
    asset_type: Literal["image", "chart", "table", "diagram", "screenshot", "generated_image"]
    source_type: Literal["input_resource", "web_search", "code_render", "image_generation"]
    uri: Optional[str] = None
    local_path: Optional[str] = None
    caption: Optional[str] = None
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class MediaAssetLibrary(ArtifactModel):
    library_id: str
    assets: Dict[str, MediaAsset] = Field(default_factory=dict)
    section_asset_ids: Dict[str, List[str]] = Field(default_factory=dict)
    visual_need_asset_ids: Dict[str, List[str]] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class VisualInstruction(BaseModel):
    need_id: str
    visual_type: Literal["image", "chart", "table", "diagram"]
    purpose: str
    preferred_strategy: Literal[
        "web_search", "kb_search", "image_generation", "code_render"
    ] = "web_search"
    required: bool = True
    meta: Dict[str, Any] = Field(default_factory=dict)
