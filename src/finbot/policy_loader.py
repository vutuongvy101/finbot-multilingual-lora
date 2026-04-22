from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyBundle:
    pii_rules: str
    refusal_topics: str
    output_rules: str


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_policies(base_dir: str | Path = "policies") -> PolicyBundle:
    base = Path(base_dir)
    return PolicyBundle(
        pii_rules=_read_text(base / "pii_rules.md"),
        refusal_topics=_read_text(base / "refusal_topics.md"),
        output_rules=_read_text(base / "output_rules.md"),
    )