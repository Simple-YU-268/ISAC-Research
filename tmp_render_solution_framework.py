from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(r"deliverables/poster_assets/solution_framework.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 2400, 1000
BG = "#FFFFFF"
NAVY = "#0B5797"
MID_BLUE = "#AFC8E3"
LINE_BLUE = "#82ACD6"
CARD = "#F3F7FB"
TEXT = "#0A568F"
MUTED = "#42617A"
FOOT = "#E8F2FB"

font_dir = Path(r"C:\Windows\Fonts")
font_regular = font_dir / "segoeui.ttf"
font_bold = font_dir / "segoeuib.ttf"

def font(size, bold=False):
    return ImageFont.truetype(str(font_bold if bold else font_regular), size)

def centered(draw, box, value, fnt, fill, spacing=8):
    x0, y0, x1, y1 = box
    bb = draw.multiline_textbbox((0, 0), value, font=fnt, spacing=spacing, align="center")
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.multiline_text((x0 + (x1-x0-tw)/2, y0 + (y1-y0-th)/2), value,
                        font=fnt, fill=fill, spacing=spacing, align="center")

def rounded(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# Header and outer frame.
rounded(d, (50, 35, 2350, 950), 92, "#FFFFFF", LINE_BLUE, 4)
rounded(d, (55, 18, 1200, 86), 18, MID_BLUE)
d.text((92, 35), "PROPOSED SOLUTION FRAMEWORK", font=font(28, True), fill="#17314A")

subtitle = "From participation-constrained MINLP to physically validated ISAC beamforming"
d.text((1220, 39), subtitle, font=font(23), fill=MUTED)

# Step cards.
left, top, card_w, card_h, gap = 100, 200, 500, 560, 55
cards = [
    ("STEP 1", "P1: Participation-Constrained\nMINLP", "Jointly optimize\nW_k, S_p, and b_mp\n\nRobust SINR • PCRB\nPer-AP power constraints"),
    ("STEP 2", "P2: Covariance Lifting\nand SDR", "Lift\nW_k = w_k w_k^H\n\nRelax rank-one constraints\nto obtain an SDP"),
    ("STEP 3", "P3: Dual DC-SCA\nOptimization", "Robust S-procedure\nPCRB Schur LMIs\n\nRank + binary DC penalties\nidentify stable support"),
    ("STEP 4", "Certified Physical\nRecovery", "Top-N support projection\nFixed-b re-optimization\n\nRank-one extraction\nand feasibility validation"),
]

for i, (tag, title, body) in enumerate(cards):
    x = left + i * (card_w + gap)
    # subtle shadow then card
    rounded(d, (x+8, top+10, x+card_w+8, top+card_h+10), 58, "#E3ECF5")
    rounded(d, (x, top, x+card_w, top+card_h), 58, CARD)
    rounded(d, (x+38, top+36, x+174, top+88), 20, NAVY)
    centered(d, (x+38, top+36, x+174, top+88), tag, font(19, True), "#FFFFFF")
    centered(d, (x+45, top+116, x+card_w-45, top+230), title, font(31, True), TEXT, 4)
    centered(d, (x+44, top+258, x+card_w-44, top+card_h-42), body, font(25, True), TEXT, 10)
    if i < len(cards)-1:
        ax = x + card_w + 10
        ay = top + card_h // 2
        d.line((ax, ay, ax+34, ay), fill=LINE_BLUE, width=8)
        d.polygon([(ax+34, ay-14), (ax+58, ay), (ax+34, ay+14)], fill=LINE_BLUE)

# Outcome band.
rounded(d, (310, 805, 2090, 895), 28, FOOT, outline=MID_BLUE, width=2)
d.text((365, 827), "OUTPUT", font=font(25, True), fill=NAVY)
d.text((535, 825), "Physically validated binary sensing clusters and robust transmit covariances",
       font=font(28, True), fill=TEXT)

img.save(OUT, optimize=True)
print(OUT.resolve())
