"""Category-aware CC decision engine."""
import logging

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Combine audio confidence + visual reaction score to make CC/no-CC decisions.

    Key design choices:
    1. Category-aware weights — explosions weighted differently from doorbells
    2. Speech-pause bonus — if speech stopped before the event, it's significant
    3. Scene-cut fallback — on scene cut, use audio-only (visual unreliable)
    4. Max CC duration — split long events into <=3s chunks (subtitle standard)
    """

    def __init__(self, config: dict, category_mapper):
        """
        Args:
            config: Full config dict from default.yaml.
            category_mapper: CategoryMapper instance.
        """
        self.default_audio_weight = config['fusion']['audio_weight']
        self.default_visual_weight = config['fusion']['visual_weight']
        self.default_threshold = config['fusion']['threshold']
        self.speech_bonus = config['fusion']['speech_pause_bonus']
        self.max_cc_duration = config['output']['max_cc_duration']
        self.category_mapper = category_mapper

    def decide(self, events: list) -> list:
        """
        Apply category-aware fusion and make CC/no-CC decisions.

        For each event:
        1. Look up its sound category -> get weights and threshold
        2. If on scene cut -> audio-only mode (visual unreliable)
        3. Compute: combined = alpha * audio_conf + beta * reaction_score + bonus
        4. Accept if combined >= category threshold

        Args:
            events: List of event dicts with confidence, reaction_score,
                    on_scene_cut, and speech_paused fields.

        Returns:
            List of accepted events with combined_score, category, accepted fields.
        """
        for event in events:
            # Get category-specific fusion parameters
            cat = self.category_mapper.get_category(event["label"])
            event["category"] = cat["category"]

            alpha = cat["audio_weight"]
            beta = cat["visual_weight"]
            threshold = cat["threshold"]

            # Scene cut -> audio-only mode
            if event.get("on_scene_cut", False):
                combined = event["confidence"]
                # Stricter threshold since we're missing visual confirmation
                threshold = max(threshold, 0.50)
                logger.debug(f"Event '{event['label']}' on scene cut -> audio-only, "
                             f"threshold raised to {threshold:.2f}")
            else:
                combined = (alpha * event["confidence"] +
                            beta * event.get("reaction_score", 0.0))

            # Speech-pause bonus
            if event.get("speech_paused", False):
                combined += self.speech_bonus
                logger.debug(f"Event '{event['label']}': +{self.speech_bonus} speech-pause bonus")

            combined = min(combined, 1.0)
            event["combined_score"] = round(combined, 4)
            event["accepted"] = combined >= threshold

            logger.info(
                f"Event #{event['id']} '{event['label']}' [{cat['category']}]: "
                f"audio={event['confidence']:.2f} visual={event.get('reaction_score', 0):.2f} "
                f"combined={combined:.2f} thresh={threshold:.2f} -> "
                f"{'ACCEPT' if event['accepted'] else 'REJECT'}"
            )

        # Filter to accepted only
        accepted = [e for e in events if e["accepted"]]
        logger.info(f"Accepted {len(accepted)} / {len(events)} events")

        # Split long events into <=3s chunks
        accepted = self._split_long_events(accepted)

        return accepted

    def _split_long_events(self, events: list) -> list:
        """
        Split events longer than max_cc_duration into chunks.
        Subtitle standard: no single CC should exceed 3 seconds.
        """
        result = []
        for event in events:
            duration = event["end_time"] - event["start_time"]
            if duration <= self.max_cc_duration:
                result.append(event)
            else:
                t = event["start_time"]
                chunk_id = 0
                while t < event["end_time"]:
                    chunk_end = min(t + self.max_cc_duration, event["end_time"])
                    chunk = event.copy()
                    chunk["start_time"] = round(t, 3)
                    chunk["end_time"] = round(chunk_end, 3)
                    chunk["id"] = f"{event['id']}_{chunk_id}"
                    result.append(chunk)
                    t = chunk_end
                    chunk_id += 1

        return result
