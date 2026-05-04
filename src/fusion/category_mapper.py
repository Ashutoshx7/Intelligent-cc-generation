"""Map YAMNet class labels to behavioral sound categories."""
import yaml
import logging

logger = logging.getLogger(__name__)


class CategoryMapper:
    """
    Maps YAMNet's 521 class names to behavioral categories:
    - high_impact: caption even without visual reaction (explosion, gunshot)
    - interactive: only caption if someone reacts (doorbell, knock)
    - social: human non-speech, context-dependent (laughter, applause)
    - ambient: almost never caption (music, rain, traffic)
    - default: fallback for unmapped classes

    Each category has its own audio_weight, visual_weight, and threshold
    for the fusion engine.
    """

    def __init__(self, categories_path: str = "config/sound_categories.yaml"):
        self.lookup, self.default = self._load(categories_path)
        logger.info(f"Loaded {len(self.lookup)} class->category mappings")

    def _load(self, path: str) -> tuple:
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        lookup = {}
        for cat_name, cat_data in raw.items():
            if cat_name == "default":
                continue
            for class_name in cat_data.get("classes", []):
                lookup[class_name] = {
                    "category": cat_name,
                    "audio_weight": cat_data["audio_weight"],
                    "visual_weight": cat_data["visual_weight"],
                    "threshold": cat_data["threshold"],
                }

        default_data = raw.get("default", {
            "audio_weight": 0.6,
            "visual_weight": 0.4,
            "threshold": 0.45,
        })
        default_data["category"] = "default"

        return lookup, default_data

    def get_category(self, label: str) -> dict:
        """
        Get fusion parameters for a YAMNet class label.

        Tries exact match first, then substring match (YAMNet labels
        can be compound like "Gunshot, gunfire").

        Returns:
            {"category": str, "audio_weight": float,
             "visual_weight": float, "threshold": float}
        """
        # Exact match
        if label in self.lookup:
            return self.lookup[label]

        # Substring match
        for class_name, params in self.lookup.items():
            if class_name.lower() in label.lower() or label.lower() in class_name.lower():
                return params

        # Fallback
        return self.default
