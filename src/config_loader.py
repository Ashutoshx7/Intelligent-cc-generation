"""Load and merge YAML config files."""
import yaml


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Load main config YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sound_categories(path: str = "config/sound_categories.yaml") -> tuple:
    """
    Load sound categories and build a class→category lookup.

    Returns:
        (lookup_dict, default_dict)
        lookup_dict: {"Gunshot, gunfire": {"category": "high_impact", "audio_weight": 0.85, ...}}
        default_dict: {"audio_weight": 0.6, "visual_weight": 0.4, "threshold": 0.45}
    """
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    lookup = {}
    for category_name, category_data in raw.items():
        if category_name == "default":
            continue
        for class_name in category_data.get("classes", []):
            lookup[class_name] = {
                "category": category_name,
                "audio_weight": category_data["audio_weight"],
                "visual_weight": category_data["visual_weight"],
                "threshold": category_data["threshold"],
            }

    default = raw.get("default", {
        "audio_weight": 0.6,
        "visual_weight": 0.4,
        "threshold": 0.45,
    })
    default["category"] = "default"

    return lookup, default
