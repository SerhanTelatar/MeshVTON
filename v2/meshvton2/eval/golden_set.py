"""Golden set manifest şeması ve yükleyicisi.

Manifest repo'ya commit edilir (JSON); görüntülerin kendisi gitignored
(VITON-HD/CLOTH3D lisansları — yeniden dağıtım yok).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class GoldenPerson:
    id: str
    image: str          # golden_root'a göreli yol
    source: str         # "vitonhd_test" | "external"
    pose: str = "front"  # "front" | "side" | "three_quarter" | "back"


@dataclass(frozen=True)
class GoldenGarment:
    id: str
    mesh: str            # .obj yolu (garments_root'a göreli)
    texture: str | None  # sibling texture PNG; None = mesh materyalinden
    category: str = "top"     # "top" | "dress" | "bottom"
    pattern: str = "solid"    # "solid" | "stripe" | "logo" | "plaid" | ...


@dataclass(frozen=True)
class GoldenCombo:
    person_id: str
    garment_id: str


@dataclass
class GoldenManifest:
    root: Path
    persons: list[GoldenPerson]
    garments: list[GoldenGarment]
    angles: list[int]
    combos: list[GoldenCombo] = field(default_factory=list)

    def __post_init__(self):
        pids = {p.id for p in self.persons}
        gids = {g.id for g in self.garments}
        if len(pids) != len(self.persons):
            raise ValueError("Yinelenen person id")
        if len(gids) != len(self.garments):
            raise ValueError("Yinelenen garment id")
        if not self.combos:
            self.combos = [GoldenCombo(p.id, g.id) for p in self.persons for g in self.garments]
        for c in self.combos:
            if c.person_id not in pids or c.garment_id not in gids:
                raise ValueError(f"Combo bilinmeyen id içeriyor: {c}")

    def items(self):
        """Değerlendirilecek (person, garment, angle) üçlülerini üretir."""
        by_pid = {p.id: p for p in self.persons}
        by_gid = {g.id: g for g in self.garments}
        for c in self.combos:
            for angle in self.angles:
                yield by_pid[c.person_id], by_gid[c.garment_id], angle

    def to_dict(self) -> dict:
        return {
            "version": MANIFEST_VERSION,
            "persons": [vars(p) for p in self.persons],
            "garments": [vars(g) for g in self.garments],
            "angles": self.angles,
            "combos": [vars(c) for c in self.combos],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))


def load_manifest(path: str | Path) -> GoldenManifest:
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Manifest sürümü uyumsuz: {data.get('version')} != {MANIFEST_VERSION}")
    return GoldenManifest(
        root=path.parent,
        persons=[GoldenPerson(**p) for p in data["persons"]],
        garments=[GoldenGarment(**g) for g in data["garments"]],
        angles=list(data["angles"]),
        combos=[GoldenCombo(**c) for c in data.get("combos", [])],
    )
