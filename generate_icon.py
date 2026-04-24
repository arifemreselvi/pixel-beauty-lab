from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def make_icon(path: Path) -> None:
    size = 256
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for y in range(size):
        t = y / (size - 1)
        r = int(35 + (186 - 35) * t)
        g = int(32 + (144 - 32) * t)
        b = int(29 + (90 - 29) * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))

    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    gdraw.ellipse((34, 34, 222, 222), fill=(255, 235, 196, 82))
    glow = glow.filter(ImageFilter.GaussianBlur(12))
    canvas = Image.alpha_composite(canvas, glow)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((26, 26, 230, 230), radius=42, outline=(248, 229, 196, 225), width=6)
    draw.rounded_rectangle((62, 78, 194, 178), radius=18, outline=(252, 242, 224, 235), width=7)
    draw.rectangle((108, 54, 148, 78), fill=(252, 242, 224, 235))

    try:
        font = ImageFont.truetype("arial.ttf", 62)
    except OSError:
        font = ImageFont.load_default()

    draw.text((78, 172), "4x", fill=(252, 242, 224, 245), font=font)

    canvas.save(path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "assets" / "fusion_upscaler.ico"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_icon(out)
    print(f"Created icon: {out}")
