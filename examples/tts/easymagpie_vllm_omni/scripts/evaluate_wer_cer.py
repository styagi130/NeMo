#!/usr/bin/env python3
"""Transcribe generated EasyMagpie WAVs and report corpus WER/CER."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def load_references(path: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        uttid, text = line.split("\t", 1)
        references[uttid.strip()] = text.strip()
    return references


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    references = load_references(args.text_file)
    wav_paths = sorted(args.audio_dir.glob("*.wav"))
    if not wav_paths:
        raise ValueError(f"No WAV files found in {args.audio_dir}")

    missing = [path.stem for path in wav_paths if path.stem not in references]
    if missing:
        raise ValueError(f"Missing references for: {missing}")

    use_cuda = args.device.startswith("cuda")
    dtype = torch.float16 if use_cuda else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    ).to(args.device)
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=args.device,
    )

    normalized_references: list[str] = []
    normalized_hypotheses: list[str] = []
    samples: list[dict[str, str]] = []
    for wav_path in wav_paths:
        audio, sample_rate = sf.read(wav_path, dtype="float32")
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio in {wav_path}, got shape {audio.shape}")
        target_sample_rate = processor.feature_extractor.sampling_rate
        if sample_rate != target_sample_rate:
            divisor = math.gcd(sample_rate, target_sample_rate)
            audio = resample_poly(audio, target_sample_rate // divisor, sample_rate // divisor).astype("float32")
            sample_rate = target_sample_rate
        result = transcriber(
            {"array": audio, "sampling_rate": sample_rate},
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
        reference = references[wav_path.stem]
        hypothesis = result["text"].strip()
        normalized_reference = processor.tokenizer.normalize(reference)
        normalized_hypothesis = processor.tokenizer.normalize(hypothesis)
        normalized_references.append(normalized_reference)
        normalized_hypotheses.append(normalized_hypothesis)
        samples.append(
            {
                "uttid": wav_path.stem,
                "reference": reference,
                "hypothesis": hypothesis,
                "normalized_reference": normalized_reference,
                "normalized_hypothesis": normalized_hypothesis,
            }
        )
        print(f"{wav_path.stem}: {hypothesis}")

    word_result = jiwer.process_words(normalized_references, normalized_hypotheses)
    char_result = jiwer.process_characters(normalized_references, normalized_hypotheses)
    output = {
        "model": args.model,
        "num_samples": len(samples),
        "wer": word_result.wer,
        "cer": char_result.cer,
        "word_counts": {
            "hits": word_result.hits,
            "substitutions": word_result.substitutions,
            "deletions": word_result.deletions,
            "insertions": word_result.insertions,
        },
        "character_counts": {
            "hits": char_result.hits,
            "substitutions": char_result.substitutions,
            "deletions": char_result.deletions,
            "insertions": char_result.insertions,
        },
        "samples": samples,
    }
    print(f"WER: {100 * word_result.wer:.2f}%")
    print(f"CER: {100 * char_result.cer:.2f}%")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
