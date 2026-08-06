"""Stable project-wide title and dataset taxonomies.

Class order is part of the model/checkpoint contract.  The original ten-class
tomato taxonomy remains available for old pilot artifacts, while the thesis
pipeline uses the complete 38-class PlantVillage ``raw/color`` taxonomy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

PROJECT_VERSION = "0.1.0"

OFFICIAL_TITLE = (
    "Nghiên cứu và xây dựng hệ thống Học liên kết (Federated Learning) "
    "cho phát hiện sâu bệnh cây trồng qua ảnh trên dữ liệu phân tán, "
    "không đồng nhất giữa các cơ sở nông nghiệp."
)


@dataclass(frozen=True, slots=True)
class DatasetTaxonomy:
    """An immutable folder-to-label contract for one dataset scope."""

    scope: str
    class_names: tuple[str, ...]
    class_groups: tuple[str, ...]
    folder_to_class: dict[str, str]

    def __post_init__(self) -> None:
        if not self.scope:
            raise ValueError("taxonomy scope cannot be empty")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ValueError("taxonomy classes must be unique")
        if len(self.class_groups) != len(self.class_names):
            raise ValueError("taxonomy group count must equal class count")
        if set(self.folder_to_class.values()) != set(self.class_names):
            raise ValueError("folder mapping must cover every taxonomy class exactly")

    @property
    def healthy_class_ids(self) -> tuple[int, ...]:
        return tuple(
            index for index, group in enumerate(self.class_groups) if group == "healthy"
        )


TOMATO_CLASSES: tuple[str, ...] = (
    "Tomato healthy",
    "Bacterial spot",
    "Early blight",
    "Late blight",
    "Leaf mold",
    "Septoria leaf spot",
    "Target spot",
    "Tomato mosaic virus",
    "Tomato yellow leaf curl virus",
    "Two-spotted spider mite",
)

TOMATO_CROP_NAME = "Tomato"

TOMATO_CLASS_GROUPS: tuple[str, ...] = (
    "healthy",
    "disease",
    "disease",
    "disease",
    "disease",
    "disease",
    "disease",
    "disease",
    "disease",
    "pest",
)

PLANTVILLAGE_FOLDER_TO_CLASS: dict[str, str] = {
    "Tomato___healthy": "Tomato healthy",
    "Tomato___Bacterial_spot": "Bacterial spot",
    "Tomato___Early_blight": "Early blight",
    "Tomato___Late_blight": "Late blight",
    "Tomato___Leaf_Mold": "Leaf mold",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot",
    "Tomato___Target_Spot": "Target spot",
    "Tomato___Tomato_mosaic_virus": "Tomato mosaic virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Tomato yellow leaf curl virus"
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Two-spotted spider mite"
    ),
}

TOMATO_TAXONOMY = DatasetTaxonomy(
    scope="tomato",
    class_names=TOMATO_CLASSES,
    class_groups=TOMATO_CLASS_GROUPS,
    folder_to_class=PLANTVILLAGE_FOLDER_TO_CLASS,
)


PLANTVILLAGE_FULL_FOLDER_TO_CLASS: dict[str, str] = {
    "Apple___Apple_scab": "Apple - Apple scab",
    "Apple___Black_rot": "Apple - Black rot",
    "Apple___Cedar_apple_rust": "Apple - Cedar apple rust",
    "Apple___healthy": "Apple - healthy",
    "Blueberry___healthy": "Blueberry - healthy",
    "Cherry_(including_sour)___Powdery_mildew": (
        "Cherry (including sour) - Powdery mildew"
    ),
    "Cherry_(including_sour)___healthy": "Cherry (including sour) - healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Corn (maize) - Cercospora leaf spot / Gray leaf spot"
    ),
    "Corn_(maize)___Common_rust_": "Corn (maize) - Common rust",
    "Corn_(maize)___Northern_Leaf_Blight": (
        "Corn (maize) - Northern leaf blight"
    ),
    "Corn_(maize)___healthy": "Corn (maize) - healthy",
    "Grape___Black_rot": "Grape - Black rot",
    "Grape___Esca_(Black_Measles)": "Grape - Esca (Black Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Grape - Leaf blight (Isariopsis Leaf Spot)"
    ),
    "Grape___healthy": "Grape - healthy",
    "Orange___Haunglongbing_(Citrus_greening)": (
        "Orange - Huanglongbing (Citrus greening)"
    ),
    "Peach___Bacterial_spot": "Peach - Bacterial spot",
    "Peach___healthy": "Peach - healthy",
    "Pepper,_bell___Bacterial_spot": "Bell pepper - Bacterial spot",
    "Pepper,_bell___healthy": "Bell pepper - healthy",
    "Potato___Early_blight": "Potato - Early blight",
    "Potato___Late_blight": "Potato - Late blight",
    "Potato___healthy": "Potato - healthy",
    "Raspberry___healthy": "Raspberry - healthy",
    "Soybean___healthy": "Soybean - healthy",
    "Squash___Powdery_mildew": "Squash - Powdery mildew",
    "Strawberry___Leaf_scorch": "Strawberry - Leaf scorch",
    "Strawberry___healthy": "Strawberry - healthy",
    "Tomato___Bacterial_spot": "Tomato - Bacterial spot",
    "Tomato___Early_blight": "Tomato - Early blight",
    "Tomato___Late_blight": "Tomato - Late blight",
    "Tomato___Leaf_Mold": "Tomato - Leaf mold",
    "Tomato___Septoria_leaf_spot": "Tomato - Septoria leaf spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Tomato - Two-spotted spider mite"
    ),
    "Tomato___Target_Spot": "Tomato - Target spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Tomato - Tomato yellow leaf curl virus"
    ),
    "Tomato___Tomato_mosaic_virus": "Tomato - Tomato mosaic virus",
    "Tomato___healthy": "Tomato - healthy",
}

PLANTVILLAGE_FULL_CLASSES: tuple[str, ...] = tuple(
    PLANTVILLAGE_FULL_FOLDER_TO_CLASS.values()
)
PLANTVILLAGE_FULL_CLASS_GROUPS: tuple[str, ...] = tuple(
    "healthy"
    if folder.endswith("___healthy")
    else "pest"
    if "Spider_mites" in folder
    else "disease"
    for folder in PLANTVILLAGE_FULL_FOLDER_TO_CLASS
)
PLANTVILLAGE_FULL_TAXONOMY = DatasetTaxonomy(
    scope="plantvillage-full",
    class_names=PLANTVILLAGE_FULL_CLASSES,
    class_groups=PLANTVILLAGE_FULL_CLASS_GROUPS,
    folder_to_class=PLANTVILLAGE_FULL_FOLDER_TO_CLASS,
)


def taxonomy_from_scope(scope: str) -> DatasetTaxonomy:
    """Resolve a CLI/config scope without silently changing class order."""

    normalized = scope.strip().lower().replace("_", "-")
    if normalized in {"tomato", "tomato-mvp", "legacy"}:
        return TOMATO_TAXONOMY
    if normalized in {"plantvillage-full", "full", "plantvillage-38"}:
        return PLANTVILLAGE_FULL_TAXONOMY
    raise ValueError(f"unsupported taxonomy scope: {scope!r}")


def taxonomy_from_class_order(
    class_order: Sequence[str] | None,
) -> DatasetTaxonomy | None:
    """Identify which known taxonomy a checkpoint's class order belongs to.

    Returns ``None`` for an order that matches neither, so callers can report
    an unknown taxonomy instead of silently assuming one.
    """

    if not class_order:
        return None
    candidate = tuple(class_order)
    for taxonomy in (TOMATO_TAXONOMY, PLANTVILLAGE_FULL_TAXONOMY):
        if candidate == taxonomy.class_names:
            return taxonomy
    return None


def class_group_for_name(class_name: str) -> str:
    """Infer a stable high-level group for checkpoint-driven inference."""

    lowered = class_name.lower()
    if "healthy" in lowered:
        return "healthy"
    if "spider mite" in lowered:
        return "pest"
    return "disease"


def crop_for_class_name(class_name: str) -> str:
    """Return the crop prefix used by the full taxonomy, or legacy Tomato."""

    if " - " in class_name:
        return class_name.split(" - ", maxsplit=1)[0]
    return TOMATO_CROP_NAME
