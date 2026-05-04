"""Map YAMNet class names to human-readable CC labels."""

# Core label mappings — YAMNet class name -> CC bracket label
LABEL_MAP = {
    # High-impact
    "Gunshot, gunfire": "[gunshot]",
    "Explosion": "[explosion]",
    "Scream": "[screaming]",
    "Glass": "[glass breaking]",
    "Siren": "[siren]",
    "Alarm": "[alarm]",
    "Thunder": "[thunder]",
    "Vehicle horn, car horn, honking": "[honking]",
    "Car alarm": "[car alarm]",
    "Fire alarm": "[fire alarm]",
    "Shatter": "[shattering]",

    # Interactive
    "Doorbell": "[doorbell]",
    "Knock": "[knocking]",
    "Telephone bell ringing": "[phone ringing]",
    "Dog": "[dog barking]",
    "Cat": "[cat meowing]",
    "Whistle": "[whistle]",
    "Ringtone": "[phone ringing]",

    # Social
    "Laughter": "[laughter]",
    "Applause": "[applause]",
    "Crying, sobbing": "[crying]",
    "Crowd": "[crowd noise]",
    "Cheering": "[cheering]",
    "Cough": "[coughing]",
    "Sneeze": "[sneezing]",
    "Clapping": "[clapping]",
    "Booing": "[booing]",

    # India-specific mappings
    "Drum": "[drums]",              # covers dhol, tabla, mridangam
    "Bell": "[bell]",               # temple bells
    "Fireworks": "[firecrackers]",  # Diwali scenes
    "Splash, splatter": "[splash]",
    "Engine starting": "[engine]",
    "Squeal": "[tires screeching]",
    "Thump, thud": "[thud]",
    "Slam": "[door slam]",
    "Bang": "[bang]",
    "Crash": "[crash]",
}


def map_label(yamnet_class: str) -> str:
    """
    Convert YAMNet class name to CC-friendly bracket label.

    Tries exact match first, then checks if any key is a substring
    of the class name. Falls back to first word of class name.

    Examples:
        "Gunshot, gunfire"                -> "[gunshot]"
        "Vehicle horn, car horn, honking" -> "[honking]"
        "Unknown weird class"             -> "[unknown]"
    """
    # Exact match
    if yamnet_class in LABEL_MAP:
        return LABEL_MAP[yamnet_class]

    # Substring match (handles compound YAMNet labels)
    for key, label in LABEL_MAP.items():
        if key.lower() in yamnet_class.lower() or yamnet_class.lower() in key.lower():
            return label

    # Fallback: first word, lowercase, in brackets
    first_word = yamnet_class.split(",")[0].split(" ")[0].strip().lower()
    return f"[{first_word}]"
