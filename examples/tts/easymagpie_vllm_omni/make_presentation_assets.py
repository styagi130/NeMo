#!/usr/bin/env python3
"""Generate simple PNG diagrams for the EasyMagpie vLLM-Omni presentation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("examples/tts/easymagpie_vllm_omni/presentation_assets")
W, H = 1600, 900


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = font(42, True)
FONT_BOX = font(28, True)
FONT_TEXT = font(22)
FONT_SMALL = font(18)

COLORS = {
    "bg": "#f7f9fb",
    "ink": "#18202a",
    "muted": "#5c6670",
    "blue": "#2f6fed",
    "green": "#1f9d55",
    "orange": "#d97706",
    "purple": "#7c3aed",
    "red": "#b42318",
    "line": "#9aa7b2",
    "white": "#ffffff",
    "pale_blue": "#eaf1ff",
    "pale_green": "#e9f8ef",
    "pale_orange": "#fff3df",
    "pale_purple": "#f2ecff",
    "pale_red": "#fff0ee",
}


def canvas(title: str):
    img = Image.new("RGB", (W, H), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.text((60, 38), title, fill=COLORS["ink"], font=FONT_TITLE)
    d.line((60, 96, W - 60, 96), fill="#d5dde5", width=3)
    return img, d


def wrap_text(draw, text: str, max_width: int, fnt) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for word in words:
        test = " ".join(cur + [word])
        bbox = draw.textbbox((0, 0), test, font=fnt)
        if bbox[2] - bbox[0] <= max_width or not cur:
            cur.append(word)
        else:
            lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines


def rounded_box(draw, xy, fill, outline, title, body=None, accent=None):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    if accent:
        draw.rounded_rectangle((x1, y1, x1 + 14, y2), radius=12, fill=accent)
    draw.text((x1 + 30, y1 + 24), title, fill=COLORS["ink"], font=FONT_BOX)
    if body:
        y = y1 + 72
        for line in wrap_text(draw, body, x2 - x1 - 60, FONT_TEXT):
            draw.text((x1 + 30, y), line, fill=COLORS["muted"], font=FONT_TEXT)
            y += 32


def arrow(draw, start, end, color=COLORS["line"], width=5):
    draw.line((start, end), fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - 22 * direction, y2 - 12), (x2 - 22 * direction, y2 + 12)]
    else:
        direction = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - 12, y2 - 22 * direction), (x2 + 12, y2 - 22 * direction)]
    draw.polygon(pts, fill=color)


def save(img, name):
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / name, quality=95)


def arch_overview():
    img, d = canvas("Runtime Architecture")
    rounded_box(
        d,
        (80, 170, 400, 330),
        COLORS["pale_orange"],
        COLORS["orange"],
        "Client",
        "Whole-text or streaming-text gRPC request.",
        COLORS["orange"],
    )
    rounded_box(
        d,
        (510, 145, 840, 355),
        COLORS["pale_purple"],
        COLORS["purple"],
        "Triton Python Backend",
        "Request parsing, prompt construction, decoupled response streaming, BLS codec calls.",
        COLORS["purple"],
    )
    rounded_box(
        d,
        (950, 145, 1280, 355),
        COLORS["pale_blue"],
        COLORS["blue"],
        "vLLM-Omni",
        "EasyMagpie talker: Nemotron-H backbone plus local transformer acoustic head.",
        COLORS["blue"],
    )
    rounded_box(
        d,
        (950, 515, 1280, 705),
        COLORS["pale_green"],
        COLORS["green"],
        "TensorRT Codec",
        "Decodes stacked audio code chunks to 22.05 kHz waveform chunks.",
        COLORS["green"],
    )
    rounded_box(
        d,
        (80, 535, 400, 690),
        COLORS["white"],
        COLORS["line"],
        "Audio Stream",
        "Decoupled Triton responses return audio chunks as they are decoded.",
        COLORS["line"],
    )
    arrow(d, (400, 250), (510, 250), COLORS["orange"])
    arrow(d, (840, 250), (950, 250), COLORS["purple"])
    arrow(d, (1115, 355), (1115, 515), COLORS["blue"])
    arrow(d, (950, 610), (400, 610), COLORS["green"])
    d.text((560, 392), "model_repository/easymp", fill=COLORS["muted"], font=FONT_SMALL)
    d.text((1005, 742), "model_repository/codec/1/model.plan", fill=COLORS["muted"], font=FONT_SMALL)
    save(img, "runtime_architecture.png")


def offline_flow():
    img, d = canvas("Offline Artifact Preparation")
    boxes = [
        ((70, 170, 360, 330), COLORS["pale_orange"], COLORS["orange"], "NeMo Artifacts", ".nemo checkpoint, spectral codec .nemo, phoneme tokenizer, context audio."),
        ((475, 150, 790, 350), COLORS["pale_blue"], COLORS["blue"], "Model Conversion", "Export vLLM config, tokenizer, model.safetensors, text embedding table."),
        ((910, 150, 1225, 350), COLORS["pale_green"], COLORS["green"], "Codec Export", "Export codec decoder ONNX, then build TensorRT engine with trtexec."),
        ((475, 500, 790, 700), COLORS["pale_purple"], COLORS["purple"], "Speaker Bake", "Encode 5s reference audio into speaker_embeddings/eng.pt."),
        ((910, 500, 1225, 700), COLORS["white"], COLORS["line"], "Triton Repo", "easymp Python backend plus codec TensorRT model are served together."),
    ]
    for xy, fill, outline, title, body in boxes:
        rounded_box(d, xy, fill, outline, title, body, outline)
    arrow(d, (360, 250), (475, 250), COLORS["orange"])
    arrow(d, (790, 250), (910, 250), COLORS["blue"])
    arrow(d, (360, 250), (475, 600), COLORS["orange"])
    arrow(d, (790, 600), (910, 600), COLORS["purple"])
    arrow(d, (1068, 350), (1068, 500), COLORS["green"])
    d.text((90, 760), "Generated but untracked: codec.onnx, model.plan, easymp_vllm_model/", fill=COLORS["muted"], font=FONT_TEXT)
    save(img, "offline_artifact_flow.png")


def inference_pipeline():
    img, d = canvas("Inference Pipeline")
    y = 205
    xs = [70, 320, 570, 820, 1070, 1320]
    labels = [
        ("Text Request", "text, speaker=eng, context=[EN]", COLORS["orange"], COLORS["pale_orange"]),
        ("Prompt Build", "tokenize text, load speaker length, seed streams", COLORS["purple"], COLORS["pale_purple"]),
        ("vLLM Prefill", "speaker context + text context embeddings", COLORS["blue"], COLORS["pale_blue"]),
        ("AR Decode", "audio codebooks + phoneme/text streams", COLORS["blue"], COLORS["pale_blue"]),
        ("Codec BLS", "15-frame code chunks to TensorRT", COLORS["green"], COLORS["pale_green"]),
        ("Audio Chunks", "22.05 kHz streamed responses", COLORS["line"], COLORS["white"]),
    ]
    for i, (title, body, outline, fill) in enumerate(labels):
        rounded_box(d, (xs[i], y, xs[i] + 210, y + 250), fill, outline, title, body, outline)
        if i < len(labels) - 1:
            arrow(d, (xs[i] + 210, y + 125), (xs[i + 1], y + 125), outline)
    d.text((80, 545), "Key runtime split:", fill=COLORS["ink"], font=FONT_BOX)
    d.text(
        (80, 592),
        "vLLM-Omni handles autoregressive acoustic-code generation; Triton BLS batches codec decode calls and streams waveform chunks.",
        fill=COLORS["muted"],
        font=FONT_TEXT,
    )
    d.text((80, 655), "Default speaker path: speaker_id='eng' -> speaker_embeddings/eng.pt -> GPU-resident known speaker embedding.", fill=COLORS["muted"], font=FONT_TEXT)
    save(img, "inference_pipeline.png")


def timing_snapshot():
    img, d = canvas("Validation Snapshot")
    rounded_box(d, (90, 155, 470, 350), COLORS["pale_green"], COLORS["green"], "Triton Load", "codec READY, easymp READY, HTTP health returned 200.", COLORS["green"])
    rounded_box(d, (610, 155, 990, 350), COLORS["pale_blue"], COLORS["blue"], "E2E Request", "Hello world. -> 5 audio chunks, 0.88s WAV at 22.05 kHz.", COLORS["blue"])
    rounded_box(d, (1130, 155, 1510, 350), COLORS["pale_orange"], COLORS["orange"], "Cold Timing", "TTFA 25.59s; cold RTFX about 0.03x; chunk playback/fetch about 18.5x.", COLORS["orange"])
    d.text((95, 455), "Cold first request breakdown", fill=COLORS["ink"], font=FONT_BOX)
    x0, y0 = 120, 535
    total_w = 1240
    ttfa_w = int(total_w * 25.59 / 25.65)
    rem_w = total_w - ttfa_w
    d.rounded_rectangle((x0, y0, x0 + total_w, y0 + 70), radius=16, fill="#e7edf5", outline="#c4ced8", width=2)
    d.rounded_rectangle((x0, y0, x0 + ttfa_w, y0 + 70), radius=16, fill=COLORS["orange"])
    d.rectangle((x0 + ttfa_w - 6, y0, x0 + ttfa_w + rem_w, y0 + 70), fill=COLORS["green"])
    d.text((x0 + 30, y0 + 18), "TTFA / first-request startup: 25.59s", fill=COLORS["white"], font=FONT_TEXT)
    d.text((x0 + ttfa_w + 10, y0 + 18), "decode", fill=COLORS["white"], font=FONT_SMALL)
    d.text((x0, y0 + 100), "Use warmup + multiple requests for steady-state RTFX.", fill=COLORS["muted"], font=FONT_TEXT)
    save(img, "validation_snapshot.png")


def main():
    arch_overview()
    offline_flow()
    inference_pipeline()
    timing_snapshot()


if __name__ == "__main__":
    main()
