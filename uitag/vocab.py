"""Vocabulary loading and validation for VLM classification."""

import json
from dataclasses import dataclass
from pathlib import Path

VOCABS_DIR = Path(__file__).parent / "vocabs"


@dataclass
class Vocab:
    """A classification vocabulary."""

    name: str
    version: str
    description: str
    types: list[str]
    prompt_template: str
    fallback_type: str
    padding_pct: int

    def build_prompt(self) -> str:
        """Build the classification prompt with the type list expanded."""
        types_list = ", ".join(self.types)
        return self.prompt_template.replace("{types_list}", types_list)


def load_vocab(name_or_path: str) -> Vocab:
    """Load a vocabulary by built-in name or file path.

    Resolution order:
    1. If name_or_path ends with .json, treat as file path
    2. Otherwise, look up in uitag/vocabs/{name}.json
    """
    if name_or_path.endswith(".json"):
        path = Path(name_or_path)
    else:
        path = VOCABS_DIR / f"{name_or_path}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {name_or_path} (looked at {path})"
        )

    data = json.loads(path.read_text())
    vocab = Vocab(
        name=data["name"],
        version=data["version"],
        description=data["description"],
        types=data["types"],
        prompt_template=data["prompt_template"],
        fallback_type=data["fallback_type"],
        padding_pct=data.get("padding_pct", 25),
    )
    _validate(vocab)
    return vocab


def _validate(vocab: Vocab) -> None:
    """Validate vocabulary constraints. Raises ValueError on failure."""
    if not vocab.types:
        raise ValueError(f"Vocabulary '{vocab.name}': types must be a non-empty list")
    if "{types_list}" not in vocab.prompt_template:
        raise ValueError(
            f"Vocabulary '{vocab.name}': prompt_template must contain {{types_list}}"
        )
    if vocab.fallback_type not in vocab.types:
        raise ValueError(
            f"Vocabulary '{vocab.name}': fallback_type '{vocab.fallback_type}' "
            f"not in types list"
        )
    if not isinstance(vocab.padding_pct, int) or not (0 <= vocab.padding_pct <= 100):
        raise ValueError(
            f"Vocabulary '{vocab.name}': padding_pct must be an int between 0 and 100, "
            f"got {vocab.padding_pct!r}"
        )
