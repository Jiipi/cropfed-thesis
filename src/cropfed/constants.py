"""Stable project-wide constants.

Changing the official title or class order breaks report/code traceability. Treat
both values as part of the project's public contract.
"""

PROJECT_VERSION = "0.1.0"

OFFICIAL_TITLE = (
    "Nghiên cứu và xây dựng hệ thống học liên kết (Federated Learning) "
    "cho phát hiện sâu bệnh cây trồng qua ảnh trên dữ liệu phân tán, "
    "không đồng nhất giữa các cơ sở nông nghiệp."
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
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato yellow leaf curl virus",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Two-spotted spider mite",
}
