"""ms-swift ORM reward: pixel-space IoU with a small invalid-box penalty."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refcocog.grounding import iou_reward

from swift.plugin import ORM, orms


class RefCOCOgIoUReward(ORM):
    def __call__(self, completions, target_bbox, image_width, image_height, **kwargs):
        return [
            iou_reward(text, bbox, int(width), int(height))
            for text, bbox, width, height in zip(completions, target_bbox, image_width, image_height)
        ]


orms["refcocog_iou"] = RefCOCOgIoUReward
