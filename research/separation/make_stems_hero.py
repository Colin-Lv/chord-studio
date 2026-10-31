"""Frame the stems screenshot like the other README product shots (rounded, hairline, shadow)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SHOTS = Path(r"D:\ChordStudio\Dialogue\docs\screenshots")
RAW = Path(r"D:\ChordStudio\research\separation\shots\stems3.png")
RADIUS = 22
BORDER = (58, 74, 78, 255)
PAD = 56
SHADOW_ALPHA = 110
TARGET_WIDTH = 1800


def main():
    raw = Image.open(RAW).convert("RGB")
    # crop the right-hand empty strip so the grid keeps the frame
    width, height = raw.size
    crop = raw.crop((0, 0, width - 2, height - 70))
    crop = crop.resize((TARGET_WIDTH, round(crop.height * TARGET_WIDTH / crop.width)), Image.LANCZOS)

    canvas = (crop.width + 2 * PAD, crop.height + 2 * PAD)
    shadow = Image.new("RGBA", canvas, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [PAD, PAD + 12, PAD + crop.width, PAD + crop.height + 12],
        radius=RADIUS, fill=(0, 0, 0, SHADOW_ALPHA))
    shadow = shadow.filter(ImageFilter.GaussianBlur(22))

    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, crop.width - 1, crop.height - 1],
                                           radius=RADIUS, fill=255)
    frame = Image.new("RGBA", canvas, (0, 0, 0, 0))
    frame.alpha_composite(shadow)
    body = crop.convert("RGBA")
    body.putalpha(mask)
    frame.alpha_composite(body, (PAD, PAD))
    ImageDraw.Draw(frame).rounded_rectangle(
        [PAD, PAD, PAD + crop.width - 1, PAD + crop.height - 1],
        radius=RADIUS, outline=BORDER, width=2)
    target = SHOTS / "stems-hero.png"
    frame.save(target, optimize=True)
    print(f"{target.name}: {frame.size[0]}x{frame.size[1]}  {target.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
