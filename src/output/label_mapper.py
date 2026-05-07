"""Map YAMNet class names to human-readable CC labels.

Covers 70+ AudioSet classes organized by category.
India-specific mappings handle regional content sounds.
"""

# Core label mappings — YAMNet class name -> CC bracket label
LABEL_MAP = {
    # ═══════════════════════════════════════════
    # HIGH IMPACT — Always worth captioning
    # ═══════════════════════════════════════════
    "Gunshot, gunfire": "[gunshot]",
    "Machine gun": "[gunfire]",
    "Explosion": "[explosion]",
    "Boom": "[explosion]",
    "Scream": "[screaming]",
    "Shout": "[shouting]",
    "Yell": "[shouting]",
    "Glass": "[glass breaking]",
    "Shatter": "[shattering]",
    "Siren": "[siren]",
    "Civil defense siren": "[siren]",
    "Ambulance (siren)": "[ambulance siren]",
    "Police car (siren)": "[police siren]",
    "Fire engine, fire truck (siren)": "[fire truck siren]",
    "Alarm": "[alarm]",
    "Fire alarm": "[fire alarm]",
    "Smoke detector, smoke alarm": "[smoke alarm]",
    "Car alarm": "[car alarm]",
    "Burglar alarm": "[alarm]",
    "Thunder": "[thunder]",
    "Thunderstorm": "[thunderstorm]",
    "Vehicle horn, car horn, honking": "[honking]",
    "Truck horn, air horn": "[truck horn]",
    "Train horn": "[train horn]",

    # ═══════════════════════════════════════════
    # INTERACTIVE — Caption if someone reacts
    # ═══════════════════════════════════════════
    "Doorbell": "[doorbell]",
    "Knock": "[knocking]",
    "Tap": "[tapping]",
    "Telephone bell ringing": "[phone ringing]",
    "Ringtone": "[phone ringing]",
    "Telephone": "[phone ringing]",
    "Ding": "[ding]",
    "Ding-dong": "[doorbell]",
    "Dog": "[dog barking]",
    "Bark": "[dog barking]",
    "Growling": "[growling]",
    "Cat": "[cat meowing]",
    "Meow": "[cat meowing]",
    "Purr": "[cat purring]",
    "Hiss": "[hissing]",
    "Whistle": "[whistle]",
    "Whistling": "[whistling]",
    "Bird": "[bird call]",
    "Crow": "[crow cawing]",
    "Rooster": "[rooster crowing]",
    "Beep, bleep": "[beep]",
    "Buzzer": "[buzzer]",

    # ═══════════════════════════════════════════
    # SOCIAL — Context-dependent
    # ═══════════════════════════════════════════
    "Laughter": "[laughter]",
    "Baby laughter": "[baby laughing]",
    "Giggle": "[giggling]",
    "Chuckle, chortle": "[chuckling]",
    "Applause": "[applause]",
    "Clapping": "[clapping]",
    "Cheering": "[cheering]",
    "Crowd": "[crowd noise]",
    "Crying, sobbing": "[crying]",
    "Baby cry, infant cry": "[baby crying]",
    "Whimper": "[whimpering]",
    "Cough": "[coughing]",
    "Sneeze": "[sneezing]",
    "Snoring": "[snoring]",
    "Gasp": "[gasping]",
    "Sigh": "[sighing]",
    "Groan": "[groaning]",
    "Booing": "[booing]",

    # ═══════════════════════════════════════════
    # TRANSPORT & MECHANICAL
    # ═══════════════════════════════════════════
    "Engine starting": "[engine starting]",
    "Engine": "[engine]",
    "Idling": "[engine idling]",
    "Squeal": "[tires screeching]",
    "Tire squeal": "[tires screeching]",
    "Skidding": "[skidding]",
    "Car": "[car passing]",
    "Motorcycle": "[motorcycle]",
    "Helicopter": "[helicopter]",
    "Aircraft": "[aircraft overhead]",
    "Train": "[train]",
    "Subway, metro, underground": "[metro train]",

    # ═══════════════════════════════════════════
    # PHYSICAL IMPACTS
    # ═══════════════════════════════════════════
    "Thump, thud": "[thud]",
    "Slam": "[door slam]",
    "Bang": "[bang]",
    "Crash": "[crash]",
    "Smash": "[smash]",
    "Slap, smack": "[slap]",
    "Punch": "[punch]",
    "Thwack": "[hit]",
    "Splash, splatter": "[splash]",
    "Drip": "[dripping]",
    "Pour": "[pouring]",

    # ═══════════════════════════════════════════
    # INDIA-SPECIFIC — Regional content sounds
    # ═══════════════════════════════════════════
    "Drum": "[drums]",                  # dhol, tabla, mridangam
    "Drum kit": "[drumbeat]",
    "Tabla": "[tabla]",
    "Steel drum": "[steel drum]",
    "Bell": "[bell]",                   # temple/church bells
    "Church bell": "[bell]",
    "Cowbell": "[cowbell]",
    "Jingle bell": "[jingle]",
    "Chime": "[chime]",
    "Wind chime": "[wind chime]",
    "Gong": "[gong]",
    "Fireworks": "[firecrackers]",      # Diwali scenes
    "Firecracker": "[firecrackers]",
    "Flute": "[flute]",                 # bansuri
    "Harmonium": "[harmonium]",
    "Sitar": "[sitar]",
    "Tabla": "[tabla]",
    "Cymbal": "[cymbal]",
    "Tambourine": "[tambourine]",

    # ═══════════════════════════════════════════
    # NATURE & WEATHER
    # ═══════════════════════════════════════════
    "Rain": "[rain]",
    "Raindrop": "[raindrops]",
    "Rain on surface": "[rain]",
    "Wind": "[wind]",
    "Howling wind": "[howling wind]",
    "Stream": "[flowing water]",
    "Waterfall": "[waterfall]",
    "Ocean": "[ocean waves]",
    "Waves, surf": "[waves]",
}


def map_label(yamnet_class: str) -> str:
    """
    Convert YAMNet class name to CC-friendly bracket label.

    Tries exact match first, then checks if any key is a substring
    of the class name. Falls back to first word of class name.

    Examples:
        "Gunshot, gunfire"                -> "[gunshot]"
        "Vehicle horn, car horn, honking" -> "[honking]"
        "Beep, bleep"                     -> "[beep]"
        "Unknown weird class"             -> "[unknown]"
    """
    # Exact match
    if yamnet_class in LABEL_MAP:
        return LABEL_MAP[yamnet_class]

    # Substring match: only check if MAP KEY is substring of the YAMNet class
    # NOT the reverse — prevents 'Fire' from matching 'Gunshot, gunfire'
    for key, label in LABEL_MAP.items():
        if key.lower() in yamnet_class.lower():
            return label

    # Fallback: first word, lowercase, in brackets
    first_word = yamnet_class.split(",")[0].split(" ")[0].strip().lower()
    return f"[{first_word}]"
