from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "paper" / "assets"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE = font(44, True)
SUBTITLE = font(28, True)
BODY = font(25)
BODY_BOLD = font(25, True)
SMALL = font(21)
SMALL_BOLD = font(21, True)
TINY = font(18)


INK = "#172033"
MUTED = "#5b677a"
BLUE = "#dceaff"
BLUE_EDGE = "#4a6f9b"
RED = "#ffe2e2"
RED_EDGE = "#b45c5c"
GREEN = "#dff4e6"
GREEN_EDGE = "#4a8b63"
YELLOW = "#fff5c7"
YELLOW_EDGE = "#a48320"
PURPLE = "#eee3ff"
PURPLE_EDGE = "#765aa8"
GRAY = "#eef2f6"
GRAY_EDGE = "#8d99a8"


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(draw, box, text, fnt, fill=INK):
    x0, y0, x1, y1 = box
    w, h = text_size(draw, text, fnt)
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2 - 2), text, font=fnt, fill=fill)


def round_box(draw, box, fill, outline, radius=22, width=4):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill="#344256", width=5):
    draw.line([start, end], fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    import math

    angle = math.atan2(y1 - y0, x1 - x0)
    size = 18
    p1 = (x1 - size * math.cos(angle - 0.45), y1 - size * math.sin(angle - 0.45))
    p2 = (x1 - size * math.cos(angle + 0.45), y1 - size * math.sin(angle + 0.45))
    draw.polygon([end, p1, p2], fill=fill)


def draw_icon(draw, kind, cx, cy, color):
    """Small line icons drawn with primitives to avoid external assets."""
    w = 4
    if kind == "labels":
        for i, off in enumerate([-18, 0, 18]):
            box = (cx - 30, cy - 30 + off, cx + 30, cy - 6 + off)
            draw.rounded_rectangle(box, radius=6, outline=color, width=w)
            draw.line((cx - 18, cy - 18 + off, cx + 18, cy - 18 + off), fill=color, width=3)
    elif kind == "vector":
        draw.line((cx - 34, cy + 26, cx + 34, cy - 26), fill=color, width=w)
        draw.line((cx + 34, cy - 26, cx + 19, cy - 24), fill=color, width=w)
        draw.line((cx + 34, cy - 26, cx + 28, cy - 11), fill=color, width=w)
        draw.ellipse((cx - 42, cy + 18, cx - 26, cy + 34), fill=color)
        draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline=color, width=w)
        draw.ellipse((cx + 26, cy - 34, cx + 42, cy - 18), fill=color)
    elif kind == "tokenizer":
        for x in [-46, -12, 22]:
            draw.rounded_rectangle((cx + x, cy - 18, cx + x + 28, cy + 18), radius=7, outline=color, width=w)
        draw.line((cx - 58, cy + 34, cx + 58, cy + 34), fill=color, width=w)
        draw.line((cx - 34, cy + 24, cx - 34, cy + 44), fill=color, width=w)
        draw.line((cx + 2, cy + 24, cx + 2, cy + 44), fill=color, width=w)
        draw.line((cx + 38, cy + 24, cx + 38, cy + 44), fill=color, width=w)
    elif kind == "eval":
        draw.rectangle((cx - 34, cy - 32, cx + 34, cy + 32), outline=color, width=w)
        for i, h in enumerate([34, 48, 24]):
            x0 = cx - 22 + i * 22
            draw.rectangle((x0, cy + 22 - h, x0 + 10, cy + 22), fill=color)
        draw.line((cx - 22, cy + 22, cx + 24, cy + 22), fill=color, width=3)


def token(draw, x, y, label, fill, edge, pad=22, height=64, fnt=BODY, radius=18):
    w, _ = text_size(draw, label, fnt)
    box = (x, y, x + w + pad * 2, y + height)
    round_box(draw, box, fill, edge, radius=radius, width=4)
    center_text(draw, box, label, fnt)
    return box


def draw_activation_bar(draw, x, y, label, value, color):
    draw.text((x, y), label, font=SMALL_BOLD, fill=INK)
    bar = (x, y + 40, x + 260, y + 68)
    round_box(draw, bar, "#ffffff", "#c7ced8", radius=14, width=2)
    fill_w = int((bar[2] - bar[0]) * value)
    round_box(draw, (bar[0], bar[1], bar[0] + fill_w, bar[3]), color, color, radius=14, width=1)


def figure_spillover():
    img = Image.new("RGB", (1800, 300), "white")
    draw = ImageDraw.Draw(img)

    mono = font(33, True)
    token_font = font(32)

    label_x = 56
    token_x = 500
    row1_y = 54
    row2_y = 174
    token_h = 72
    row1_center = row1_y + token_h / 2
    row2_center = row2_y + token_h / 2

    for text, y_center in [("Spillover (benign):", row1_center), ("Hazard-aware:", row2_center)]:
        _, th = text_size(draw, text, mono)
        draw.text((label_x, y_center - th / 2 - 4), text, font=mono, fill="black")

    # Baseline row
    x = token_x
    y = row1_y
    for label, fill, edge in [
        ("anti", BLUE, BLUE_EDGE),
        ("poi", BLUE, BLUE_EDGE),
        ("son", BLUE, BLUE_EDGE),
        ("ing", BLUE, BLUE_EDGE),
        ("practice", BLUE, BLUE_EDGE),
        ("s", BLUE, BLUE_EDGE),
        ("bom", BLUE, BLUE_EDGE),
        ("bas", BLUE, BLUE_EDGE),
        ("tic", BLUE, BLUE_EDGE),
        ("style", BLUE, BLUE_EDGE),
    ]:
        b = token(draw, x, y, label, fill, edge, pad=16, height=token_h, fnt=token_font, radius=10)
        x = b[2] + 4

    # Alignment-aware row
    x = token_x
    y = row2_y
    for label, fill, edge in [
        ("antipoisoning", RED, RED_EDGE),
        ("practices", RED, RED_EDGE),
        ("bombastic", RED, RED_EDGE),
        ("style", RED, RED_EDGE),
    ]:
        b = token(draw, x, y, label, fill, edge, pad=22, height=token_h, fnt=token_font, radius=10)
        x = b[2] + 84

    out = ASSET_DIR / "token-spillover.jpg"
    img.save(out, quality=96, subsampling=0)


def stage_box(draw, box, title, lines, fill, edge):
    round_box(draw, box, fill, edge, radius=28, width=4)
    x0, y0, x1, _ = box
    draw.text((x0 + 34, y0 + 28), title, font=SUBTITLE, fill=INK)
    y = y0 + 92
    for line in lines:
        draw.text((x0 + 38, y), line, font=BODY, fill=INK)
        y += 42


def pill(draw, box, text, fill, edge):
    round_box(draw, box, fill, edge, radius=26, width=3)
    center_text(draw, box, text, BODY_BOLD)


def figure_method():
    img = Image.new("RGB", (2400, 1050), "white")
    draw = ImageDraw.Draw(img)

    draw.text((90, 55), "AAT overview", font=TITLE, fill=INK)
    draw.text((90, 112), "Few labels define a safety signal for tokenizer edits and adapter training.", font=BODY, fill=MUTED)

    # Column geometry
    y_top, y_bot = 235, 820
    boxes = [
        ((90, y_top, 495, y_bot), "1. Labels", PURPLE, PURPLE_EDGE),
        ((595, y_top, 1000, y_bot), "2. Direction", BLUE, BLUE_EDGE),
        ((1100, y_top, 1650, y_bot), "3. Intervention", GREEN, GREEN_EDGE),
        ((1750, y_top, 2310, y_bot), "4. Metrics", YELLOW, YELLOW_EDGE),
    ]
    for box, title, fill, edge in boxes:
        round_box(draw, box, fill, edge, radius=32, width=4)
        draw.text((box[0] + 34, box[1] + 30), title, font=SUBTITLE, fill=INK)

    draw_icon(draw, "labels", 410, 295, PURPLE_EDGE)
    draw_icon(draw, "vector", 910, 295, BLUE_EDGE)
    draw_icon(draw, "tokenizer", 1552, 295, GREEN_EDGE)
    draw_icon(draw, "eval", 2220, 295, YELLOW_EDGE)

    # Column 1
    pill(draw, (145, 360, 440, 426), "hazard", "#ffffff", GRAY_EDGE)
    pill(draw, (145, 475, 440, 541), "neutral", "#ffffff", GRAY_EDGE)
    pill(draw, (145, 590, 440, 656), "unlabeled", "#ffffff", GRAY_EDGE)

    # Column 2
    pill(draw, (665, 390, 930, 485), "probe", "#ffffff", BLUE_EDGE)
    pill(draw, (665, 560, 930, 655), "vector v", "#ffffff", BLUE_EDGE)

    # Column 3
    pill(draw, (1165, 350, 1425, 424), "BPE", "#ffffff", GREEN_EDGE)
    pill(draw, (1165, 470, 1425, 544), "SPM", "#ffffff", GREEN_EDGE)
    pill(draw, (1165, 600, 1425, 674), "LoRA", "#ffffff", GREEN_EDGE)

    # Column 4
    for i, line in enumerate(["PPL / TPC", "drift", "stability", "safety proxy"]):
        y = 360 + i * 92
        draw.ellipse((1805, y + 7, 1825, y + 27), fill=YELLOW_EDGE)
        draw.text((1848, y), line, font=BODY, fill=INK)

    # Main arrows
    arrow(draw, (495, 525), (595, 525), fill="#344256", width=6)
    arrow(draw, (1000, 525), (1100, 525), fill="#344256", width=6)
    arrow(draw, (1650, 525), (1750, 525), fill="#344256", width=6)

    # Subtle vertical divider in intervention column
    draw.line((1135, 565, 1615, 565), fill="#8fb99e", width=2)
    draw.text((1165, 718), "AAT tokenizer + adapter", font=SMALL_BOLD, fill=GREEN_EDGE)

    # Small footer, not a claim banner.
    draw.text((90, 900), "Detailed objectives and scoring are defined in the Method section.", font=SMALL, fill=MUTED)

    out = ASSET_DIR / "method_overview.png"
    img.save(out)


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    figure_spillover()
    figure_method()
    print(f"wrote {ASSET_DIR / 'token-spillover.jpg'}")
    print(f"wrote {ASSET_DIR / 'method_overview.png'}")


if __name__ == "__main__":
    main()
