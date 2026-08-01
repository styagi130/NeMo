from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent / "presentation_assets" / "pytorch_architecture"
W, H = 1920, 1080

COLORS = {
    "bg": "#f7f8fa",
    "ink": "#1f2933",
    "muted": "#52606d",
    "line": "#9aa5b1",
    "blue": "#2f80ed",
    "blue_l": "#dcebff",
    "teal": "#12a4a6",
    "teal_l": "#d7f7f5",
    "green": "#2f9e44",
    "green_l": "#dff5e6",
    "amber": "#f2994a",
    "amber_l": "#fff0d6",
    "red": "#d64545",
    "red_l": "#ffe3e3",
    "violet": "#7c3aed",
    "violet_l": "#eee7ff",
    "gray_l": "#edf0f2",
    "white": "#ffffff",
}


def font(size, bold=False):
    base = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
    path = f"{base}-Bold.ttf" if bold else f"{base}.ttf"
    return ImageFont.truetype(path, size)


F_TITLE = font(54, True)
F_SUB = font(31)
F_BOX = font(28, True)
F_BODY = font(23)
F_SMALL = font(21)
F_TINY = font(18)


def new_canvas(title, subtitle=None):
    img = Image.new("RGB", (W, H), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.text((70, 44), title, fill=COLORS["ink"], font=F_TITLE)
    if subtitle:
        d.text((74, 112), subtitle, fill=COLORS["muted"], font=F_SUB)
    d.line((70, 166, W - 70, 166), fill="#d9dee3", width=3)
    return img, d


def text_size(d, text, fnt):
    box = d.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_pixels(d, text, fnt, max_width):
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if text_size(d, trial, fnt)[0] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_box(d, xy, title, body=None, fill="#ffffff", outline="#9aa5b1", title_color=None, body_color=None):
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=8, fill=fill, outline=outline, width=3)
    title_color = title_color or COLORS["ink"]
    body_color = body_color or COLORS["muted"]
    pad = 24
    tx, ty = x1 + pad, y1 + pad
    for line in wrap_pixels(d, title, F_BOX, x2 - x1 - 2 * pad):
        d.text((tx, ty), line, fill=title_color, font=F_BOX)
        ty += 36
    if body:
        ty += 8
        for line in wrap_pixels(d, body, F_BODY, x2 - x1 - 2 * pad):
            d.text((tx, ty), line, fill=body_color, font=F_BODY)
            ty += 31


def arrow(d, start, end, color=None, width=5):
    color = color or COLORS["line"]
    x1, y1 = start
    x2, y2 = end
    d.line((x1, y1, x2, y2), fill=color, width=width)
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 18
    wing = 10
    pts = [
        (x2, y2),
        (x2 - head * ux + wing * px, y2 - head * uy + wing * py),
        (x2 - head * ux - wing * px, y2 - head * uy - wing * py),
    ]
    d.polygon(pts, fill=color)


def pill(d, xy, text, fill, outline=None, fnt=F_SMALL, color=None):
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=8, fill=fill, outline=outline or fill, width=2)
    tw, th = text_size(d, text, fnt)
    d.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 2), text, fill=color or COLORS["ink"], font=fnt)


def module_stack():
    img, d = new_canvas(
        "EasyMagpieTTS PyTorch Module Stack",
        "The inference base class owns runtime modules; the training subclass adds losses and dataloaders.",
    )

    draw_box(
        d,
        (120, 230, 1800, 930),
        "EasyMagpieTTSInferenceModel",
        "Wires the codec, embeddings, causal decoder, prediction heads, and streaming state.",
        fill=COLORS["white"],
        outline=COLORS["line"],
    )

    boxes = [
        ((180, 385, 515, 560), "Audio codec", "Waveform <-> discrete codebook tokens.", COLORS["teal_l"], COLORS["teal"]),
        ((570, 385, 905, 560), "Text encoder", "BPE token embedding plus optional char-aware subword features.", COLORS["blue_l"], COLORS["blue"]),
        ((960, 385, 1295, 560), "Phoneme path", "Embeds and predicts pronunciation tokens.", COLORS["green_l"], COLORS["green"]),
        ((1350, 385, 1685, 560), "Audio code path", "Embeds prior codes and projects hidden states to logits.", COLORS["amber_l"], COLORS["amber"]),
        ((370, 650, 860, 830), "Causal decoder", "HF CausalLM or Nemotron-H; runs on summed embeddings with KV cache.", COLORS["gray_l"], COLORS["ink"]),
        ((1060, 650, 1550, 830), "Local transformer", "Optional AR model predicts codebooks within one acoustic frame.", COLORS["violet_l"], COLORS["violet"]),
    ]
    for xy, title, body, fill, outline in boxes:
        draw_box(d, xy, title, body, fill=fill, outline=outline)

    arrow(d, (515, 475), (570, 475), COLORS["line"])
    arrow(d, (905, 475), (960, 475), COLORS["line"])
    arrow(d, (1295, 475), (1350, 475), COLORS["line"])
    arrow(d, (700, 560), (650, 650), COLORS["line"])
    arrow(d, (1130, 560), (1120, 650), COLORS["line"])
    arrow(d, (1510, 560), (1330, 650), COLORS["line"])
    arrow(d, (860, 740), (1060, 740), COLORS["line"])

    pill(d, (1130, 885, 1730, 935), "EasyMagpieTTSModel adds loaders, losses, and metrics", COLORS["red_l"], COLORS["red"], F_SMALL)
    img.save(OUT_DIR / "module_stack.png")


def representation_flow():
    img, d = new_canvas(
        "Representation Flow",
        "Inputs are converted into aligned embedding streams before the causal decoder sees them.",
    )
    y = 260
    left = [
        ((95, y, 380, y + 130), "Text transcript", "BPE tokens + EOS", COLORS["blue_l"], COLORS["blue"]),
        ((95, y + 180, 380, y + 310), "Context text", "speaker/style text", COLORS["blue_l"], COLORS["blue"]),
        ((95, y + 360, 380, y + 490), "Context audio", "reference waveform", COLORS["teal_l"], COLORS["teal"]),
        ((95, y + 540, 380, y + 670), "Target audio", "training only", COLORS["amber_l"], COLORS["amber"]),
    ]
    for xy, title, body, fill, outline in left:
        draw_box(d, xy, title, body, fill=fill, outline=outline)

    middle = [
        ((520, y, 865, y + 130), "embed text", "decoder embedding + optional CAS", COLORS["white"], COLORS["blue"]),
        ((520, y + 180, 865, y + 310), "context text embedding", "conditioning sequence", COLORS["white"], COLORS["blue"]),
        ((520, y + 360, 865, y + 490), "audio to codes", "discrete codebooks", COLORS["white"], COLORS["teal"]),
        ((520, y + 540, 865, y + 670), "BOS/EOS + stack", "teacher-forced input/target split", COLORS["white"], COLORS["amber"]),
    ]
    for xy, title, body, fill, outline in middle:
        draw_box(d, xy, title, body, fill=fill, outline=outline)

    draw_box(
        d,
        (1060, 315, 1450, 560),
        "Temporal alignment",
        "Context first, then text, phoneme, and audio channels are delayed and summed at matching time indices.",
        fill=COLORS["gray_l"],
        outline=COLORS["line"],
    )
    draw_box(
        d,
        (1530, 395, 1835, 500),
        "full_embedding",
        "(B, T_total, E)",
        fill=COLORS["white"],
        outline=COLORS["ink"],
    )
    draw_box(
        d,
        (1530, 570, 1835, 700),
        "Causal decoder",
        "hidden states -> heads",
        fill=COLORS["white"],
        outline=COLORS["ink"],
    )

    for yy in [y + 65, y + 245, y + 425, y + 605]:
        arrow(d, (380, yy), (520, yy), COLORS["line"])
        arrow(d, (865, yy), (1060, 440), COLORS["line"])
    arrow(d, (1450, 440), (1530, 445), COLORS["line"])
    arrow(d, (1682, 500), (1682, 570), COLORS["line"])
    img.save(OUT_DIR / "representation_flow.png")


def delayed_streams():
    img, d = new_canvas(
        "Delayed Multi-Stream Alignment",
        "Streaming mode trains the model to read text before it must emit phonemes and audio codes.",
    )
    x0, y0 = 260, 250
    cell_w, cell_h = 106, 88
    labels = ["ctx0", "ctx1", "txt0", "txt1", "txt2", "txt3", "txt4", "txt5", "txt6", "txt7", "EOS"]
    d.text((70, y0 + 24), "Decoder", fill=COLORS["ink"], font=F_BOX)
    for i, lab in enumerate(labels):
        x = x0 + i * cell_w
        d.rectangle((x, y0, x + cell_w - 6, y0 + cell_h), fill=COLORS["white"], outline=COLORS["line"], width=2)
        tw, th = text_size(d, lab, F_SMALL)
        d.text((x + (cell_w - 6 - tw) / 2, y0 + 32), lab, fill=COLORS["ink"], font=F_SMALL)

    rows = [
        ("Context", 0, 2, COLORS["teal_l"], COLORS["teal"], "context conditioning"),
        ("Text", 2, 9, COLORS["blue_l"], COLORS["blue"], "main transcript"),
        ("Phoneme", 5, 11, COLORS["green_l"], COLORS["green"], "after phoneme delay"),
        ("Audio", 7, 11, COLORS["amber_l"], COLORS["amber"], "after speech delay"),
    ]
    for r, (name, start, end, fill, outline, note) in enumerate(rows):
        y = y0 + 155 + r * 145
        for i in range(len(labels)):
            x = x0 + i * cell_w
            active = start <= i < end
            d.rectangle(
                (x, y, x + cell_w - 6, y + cell_h),
                fill=fill if active else "#e6e9ed",
                outline=outline if active else "#c8d0d8",
                width=2,
            )
            if active:
                t = "input" if name in {"Context", "Text"} else "predict"
                if name == "Audio" and i == start:
                    t = "BOS"
                tw, th = text_size(d, t, F_TINY)
                d.text((x + (cell_w - 6 - tw) / 2, y + 34), t, fill=COLORS["ink"], font=F_TINY)
        d.text((70, y + 22), name, fill=COLORS["ink"], font=F_BOX)
        d.text((x0 + len(labels) * cell_w + 40, y + 28), note, fill=COLORS["muted"], font=F_BODY)

    pill(d, (520, 990, 1400, 1044), "Conceptual example: exact delays come from TrainingMode", COLORS["white"], COLORS["line"], F_SMALL)
    img.save(OUT_DIR / "delayed_streams.png")


def streaming_loop():
    img, d = new_canvas(
        "PyTorch Inference Pipeline",
        "do_tts wraps batch construction, streaming state initialization, iterative decoding, and codec decode.",
    )
    nodes = [
        ((90, 330, 375, 500), "do_tts", "tokenize text, load context audio, build batch", COLORS["blue_l"], COLORS["blue"]),
        ((455, 330, 740, 500), "streaming init", "prepare context and prime the KV cache", COLORS["teal_l"], COLORS["teal"]),
        ((820, 280, 1160, 550), "streaming step loop", "choose phase, compose next input, run decoder, update state", COLORS["white"], COLORS["ink"]),
        ((1240, 330, 1525, 500), "streaming finalize", "slice generated codes and phonemes", COLORS["green_l"], COLORS["green"]),
        ((1605, 330, 1870, 500), "codec decode", "codes_to_audio returns waveform", COLORS["amber_l"], COLORS["amber"]),
    ]
    for xy, title, body, fill, outline in nodes:
        draw_box(d, xy, title, body, fill=fill, outline=outline)
    for a, b in [((375, 415), (455, 415)), ((740, 415), (820, 415)), ((1160, 415), (1240, 415)), ((1525, 415), (1605, 415))]:
        arrow(d, a, b, COLORS["line"])

    draw_box(
        d,
        (820, 650, 1160, 850),
        "StreamingState",
        "KV cache, positions, counters, flags, predictions",
        fill=COLORS["gray_l"],
        outline=COLORS["line"],
    )
    arrow(d, (990, 550), (990, 650), COLORS["line"])
    arrow(d, (820, 750), (720, 555), COLORS["line"])
    d.arc((705, 485, 1260, 850), start=110, end=345, fill=COLORS["line"], width=4)
    arrow(d, (1242, 725), (1158, 615), COLORS["line"])

    pill(d, (515, 905, 1405, 960), "Loop condition: until every item sees audio EOS or max_decoder_steps is reached", COLORS["white"], COLORS["line"], F_SMALL)
    img.save(OUT_DIR / "streaming_loop.png")


def local_transformer():
    img, d = new_canvas(
        "Local Transformer: Predict Codebooks Inside One Frame",
        "The main decoder emits one hidden state; the local transformer expands it into multiple audio-code tokens.",
    )
    draw_box(d, (110, 360, 420, 520), "Decoder hidden", "h_t\n(B, hidden_dim)", fill=COLORS["gray_l"], outline=COLORS["ink"])
    arrow(d, (420, 440), (545, 440), COLORS["line"])
    draw_box(d, (545, 315, 900, 565), "Local transformer input", "[h_t, previous codebook embeddings...]\nAR mode grows this sequence one codebook at a time.", fill=COLORS["violet_l"], outline=COLORS["violet"])
    arrow(d, (900, 440), (1015, 440), COLORS["line"])

    x = 1015
    y = 260
    for i in range(6):
        draw_box(
            d,
            (x + i * 130, y + (i % 2) * 160, x + i * 130 + 105, y + (i % 2) * 160 + 120),
            f"CB{i}",
            "logits",
            fill=COLORS["amber_l"],
            outline=COLORS["amber"],
        )
        if i > 0:
            arrow(d, (x + (i - 1) * 130 + 105, y + ((i - 1) % 2) * 160 + 60), (x + i * 130, y + (i % 2) * 160 + 60), COLORS["line"], 3)
    draw_box(d, (1210, 675, 1695, 835), "Frame output", "Tokens reshape to (B, codebooks, stack), then append and decode.", fill=COLORS["white"], outline=COLORS["line"])
    arrow(d, (1390, 540), (1455, 675), COLORS["line"])

    pill(d, (250, 880, 1670, 945), "Long-range decoder handles timing; local transformer handles codebook dependencies inside one frame.", COLORS["white"], COLORS["line"], F_SMALL)
    img.save(OUT_DIR / "local_transformer.png")


def training_view():
    img, d = new_canvas(
        "Training View",
        "The training subclass uses teacher-forced channels and supervises the same heads used at inference.",
    )
    draw_box(d, (95, 285, 420, 450), "Batch", "text, context, audio codes, phonemes", fill=COLORS["white"], outline=COLORS["line"])
    arrow(d, (420, 370), (520, 370), COLORS["line"])
    draw_box(d, (520, 245, 890, 500), "process_batch", "prepare context, align delays, sum streams, run decoder", fill=COLORS["blue_l"], outline=COLORS["blue"])
    arrow(d, (890, 370), (1010, 370), COLORS["line"])
    draw_box(d, (1010, 245, 1370, 500), "Prediction heads", "audio logits, phoneme logits, optional local-transformer logits", fill=COLORS["teal_l"], outline=COLORS["teal"])
    arrow(d, (1370, 370), (1490, 370), COLORS["line"])
    draw_box(d, (1490, 245, 1825, 500), "Loss", "weighted sum of codebook CE, phoneme CE, local transformer CE", fill=COLORS["red_l"], outline=COLORS["red"])

    draw_box(d, (240, 650, 590, 820), "Regularization", "text dropout, CFG dropout, phoneme corruption", fill=COLORS["gray_l"], outline=COLORS["line"])
    draw_box(d, (790, 650, 1135, 820), "Multi-mode training", "TrainingMode chooses full/streaming delays", fill=COLORS["gray_l"], outline=COLORS["line"])
    draw_box(d, (1335, 650, 1680, 820), "Validation extras", "ASR, speaker verification, UTMOS, audio examples", fill=COLORS["gray_l"], outline=COLORS["line"])
    arrow(d, (410, 650), (615, 500), COLORS["line"], 3)
    arrow(d, (965, 650), (700, 500), COLORS["line"], 3)
    arrow(d, (1505, 650), (1180, 500), COLORS["line"], 3)

    img.save(OUT_DIR / "training_view.png")


def codec_tokens():
    img, d = new_canvas(
        "Audio Is Modeled as Discrete Tokens",
        "The neural codec turns waveform into multiple codebook streams; EasyMagpie predicts those token IDs.",
    )
    draw_box(d, (115, 340, 390, 520), "Waveform", "samples at codec sample rate", fill=COLORS["teal_l"], outline=COLORS["teal"])
    arrow(d, (390, 430), (510, 430), COLORS["line"])
    draw_box(d, (510, 315, 810, 545), "Codec encoder", "encode", fill=COLORS["white"], outline=COLORS["teal"])
    arrow(d, (810, 430), (930, 430), COLORS["line"])

    x0, y0 = 930, 255
    for r, name in enumerate(["codebook 0", "codebook 1", "codebook 2", "codebook ..."]):
        y = y0 + r * 95
        d.text((x0, y + 21), name, fill=COLORS["ink"], font=F_SMALL)
        for i in range(7):
            fill = COLORS["amber_l"] if i not in {0, 6} else COLORS["gray_l"]
            d.rounded_rectangle((x0 + 165 + i * 72, y, x0 + 222 + i * 72, y + 62), radius=8, fill=fill, outline=COLORS["amber"], width=2)
            label = "BOS" if i == 0 else ("EOS" if i == 6 else str(100 + r * 10 + i))
            tw, th = text_size(d, label, F_TINY)
            d.text((x0 + 165 + i * 72 + (57 - tw) / 2, y + 19), label, fill=COLORS["ink"], font=F_TINY)

    draw_box(d, (1020, 690, 1435, 845), "Special audio tokens", "BOS/EOS plus context and mask tokens are appended after codec IDs.", fill=COLORS["white"], outline=COLORS["line"])
    arrow(d, (1435, 765), (1575, 765), COLORS["line"])
    draw_box(d, (1575, 690, 1830, 845), "Codec decoder", "codes_to_audio", fill=COLORS["teal_l"], outline=COLORS["teal"])
    img.save(OUT_DIR / "codec_tokens.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    module_stack()
    representation_flow()
    codec_tokens()
    delayed_streams()
    streaming_loop()
    local_transformer()
    training_view()
    print(f"Wrote diagrams to {OUT_DIR}")


if __name__ == "__main__":
    main()
