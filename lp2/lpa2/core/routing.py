from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

# Tags -> UNet module paths. Adjust if your diffusers version differs.
LAYER_TAGS: Dict[str, List[str]] = {
    "down1": ["down_blocks.0.attentions.0", "down_blocks.0.attentions.1"],
    "down2": ["down_blocks.1.attentions.0", "down_blocks.1.attentions.1"],
    "mid":   ["mid_block.attentions.0"],
    "up1":   ["up_blocks.1.attentions.0", "up_blocks.1.attentions.1"],
    "up2":   ["up_blocks.2.attentions.0", "up_blocks.2.attentions.1"],
}

@dataclass(frozen=True)
class RoutingPreset:
    style_tags: List[str]
    content_tags: List[str]

PRESETS: Dict[str, RoutingPreset] = {
    "vanilla": RoutingPreset([], []),
    "lpa_early_style_late": RoutingPreset(["up1","up2"], ["down1","down2","mid"]),
    "lpa_mid_only": RoutingPreset(["mid"], ["down2","up1"]),
    "lpa_late_only": RoutingPreset(["up2"], ["down1","mid"]),
    "lpa_full": RoutingPreset(["down1","down2","mid","up1","up2"],
                              ["down1","down2","mid","up1","up2"]),
}

def layers_from_preset(preset: RoutingPreset) -> dict:
    def expand(tags: List[str]) -> List[str]:
        out: List[str] = []
        for t in tags:
            out += LAYER_TAGS.get(t, [])
        return out
    return {"style_layers": expand(preset.style_tags), "content_layers": expand(preset.content_tags)}
