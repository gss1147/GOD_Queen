import random
from datetime import datetime
from typing import Dict, Any, List


def daily_spiritual_message(intent: str | None = None) -> Dict[str, Any]:
    """
    Simple, offline spiritual / motivational message generator.
    """
    base_messages = [
        "You are loved more than you know.",
        "Today is a chance to forgive yourself and begin again.",
        "The universe does not ask you to be perfect, only present.",
        "Honor your limits, but do not underestimate your strength.",
        "Small acts of kindness are sacred.",
        "Your story is still being written—stay open.",
    ]
    if intent:
        intent = intent.strip().lower()
        if "fear" in intent:
            base_messages.append("Fear is loud, but truth is steady. Breathe and act from truth.")
        if "love" in intent:
            base_messages.append("Everything you give in love reshapes the world in ways you may never see.")
        if "purpose" in intent:
            base_messages.append("Purpose is not found, it is made—one honest step at a time.")

    message = random.choice(base_messages)
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "intent": intent or "",
        "message": message,
    }


_TAROT_ARCANA: List[Dict[str, str]] = [
    {"name": "The Fool", "theme": "beginnings, trust, stepping into the unknown"},
    {"name": "The Magician", "theme": "power, focused will, using your tools"},
    {"name": "The High Priestess", "theme": "intuition, inner knowing, mystery"},
    {"name": "The Empress", "theme": "abundance, care, creativity"},
    {"name": "The Emperor", "theme": "structure, responsibility, leadership"},
    {"name": "The Hierophant", "theme": "tradition, learning, spiritual guidance"},
    {"name": "The Lovers", "theme": "connection, choice, values in relationship"},
    {"name": "The Chariot", "theme": "forward motion, discipline, victory"},
    {"name": "Strength", "theme": "courage, compassion, inner power"},
    {"name": "The Hermit", "theme": "solitude, wisdom, inner search"},
    {"name": "Wheel of Fortune", "theme": "cycles, change, fate"},
    {"name": "Justice", "theme": "truth, balance, consequences"},
    {"name": "The Hanged Man", "theme": "surrender, new perspective"},
    {"name": "Death", "theme": "transformation, endings, rebirth"},
    {"name": "Temperance", "theme": "moderation, alchemy, patience"},
    {"name": "The Devil", "theme": "attachment, illusions, temptation"},
    {"name": "The Tower", "theme": "sudden change, revelation, upheaval"},
    {"name": "The Star", "theme": "hope, healing, guidance"},
    {"name": "The Moon", "theme": "uncertainty, dreams, subconscious"},
    {"name": "The Sun", "theme": "joy, clarity, vitality"},
    {"name": "Judgement", "theme": "awakening, calling, reckoning"},
    {"name": "The World", "theme": "completion, integration, wholeness"},
]


def tarot_reading(question: str) -> Dict[str, Any]:
    """
    One-card tarot style reading.
    """
    card = random.choice(_TAROT_ARCANA)
    explanation = (
        f"Card drawn: {card['name']} — theme: {card['theme']}.\n\n"
        f"Question: {question}\n\n"
        "Interpret this as guidance, not prediction. Focus on what you can choose."
    )
    return {
        "question": question,
        "card": card["name"],
        "theme": card["theme"],
        "reading": explanation,
        "timestamp": datetime.utcnow().isoformat(),
    }