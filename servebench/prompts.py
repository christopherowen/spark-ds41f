"""Prompt sets. ``default``: 256 prompts, sixteen task styles (prose, code, structured
output, reasoning, dialogue) crossed with 32 topics, so routing, draft acceptance and
output length vary across the set the way they do across users rather than repeating a
handful of texts. The set is fixed and hashed; the run's seed only orders it."""

import hashlib
import json
import random
import sysconfig
from pathlib import Path

TEMPLATES = (
    "Explain {topic} to a curious teenager, with one concrete example.",
    "Write a short story in which {topic} changes the main character's day.",
    "Write a Python function related to {topic}, with type hints, a docstring and two tests.",
    "Compare two different approaches to {topic} and say when you would choose each.",
    "Give a step-by-step plan for a beginner who wants to get started with {topic}.",
    "Summarize the history of {topic} in a few paragraphs.",
    "Write a dialogue between a skeptic and an expert about {topic}.",
    "Return a JSON object with five common misconceptions about {topic}, each with a correction.",
    "List the ten most important terms someone should know about {topic}, with one-line definitions.",
    "Write a letter to a friend recommending they learn about {topic}, and explain why.",
    "Design a one-week lesson plan that teaches {topic}, day by day.",
    "Describe a realistic problem involving {topic} and work through how to solve it.",
    "Write a product description for an imaginary gadget built around {topic}.",
    "Explain the trade-offs and risks involved in {topic}, with numbers where you can.",
    "Write a SQL schema and three example queries for an app that tracks {topic}.",
    "Argue for and then against the claim that {topic} matters more than people think.",
)

TOPICS = (
    "hash maps", "photosynthesis", "the printing press", "public-key cryptography",
    "sourdough baking", "plate tectonics", "jazz improvisation", "compound interest",
    "rainwater harvesting", "the immune system", "chess openings", "solar panels",
    "beekeeping", "garbage collection in programming languages", "ocean tides",
    "urban gardening", "the Roman road network", "machine translation", "volcanoes",
    "budgeting for a household", "bicycle maintenance", "the water cycle",
    "database indexing", "migratory birds", "learning a musical instrument as an adult",
    "electric grids", "coffee roasting", "map projections", "earthquake-resistant buildings",
    "open-source licensing", "sleep and memory", "the history of timekeeping",
)


def default_prompts() -> list[dict]:
    """256 distinct prompts: template i % 16 with topic (i // 16 + 7 * (i % 16)) % 32, so
    every template meets 16 topics and neighbours differ in both."""
    out = []
    for i in range(len(TEMPLATES) * len(TOPICS) // 2):
        template = TEMPLATES[i % len(TEMPLATES)]
        topic = TOPICS[(i // len(TEMPLATES) + 7 * (i % len(TEMPLATES))) % len(TOPICS)]
        out.append({"id": f"d{i:03d}", "text": template.format(topic=topic)})
    return out


SETS = {"default": default_prompts}

# Python standard-library source averages about 3.8 characters per token on a DeepSeek
# tokenizer; documents are calibrated against the server's tokenizer when it has one.
SOURCE_CHARS_PER_TOKEN = 3.8


def standard_library_sources() -> list[Path]:
    root = Path(sysconfig.get_paths()["stdlib"])
    return sorted(path for path in root.glob("*.py") if path.stat().st_size > 4096)


def source_text(tokens: int, seed: int, chars_per_token: float = SOURCE_CHARS_PER_TOKEN) -> str:
    """A long real-text document: the Python standard library's source read from a
    seeded file onward. The first line makes every document's prefix its own."""
    files = standard_library_sources()
    index = random.Random(seed).randrange(len(files))
    chars = int(tokens * chars_per_token)
    parts, size = [], 0
    while size < chars:
        text = files[index % len(files)].read_text(encoding="utf-8", errors="replace")
        parts.append(text)
        size += len(text)
        index += 1
    return f"Document {seed}:\n" + "".join(parts)[:chars]


def load(name: str) -> list[dict]:
    """A named set, or a JSONL file of {"id", "text"} objects."""
    if name in SETS:
        return SETS[name]()
    with open(name, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(prompts: list[dict]) -> str:
    return hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
