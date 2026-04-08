import csv
import io
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from scipy.signal import butter, filtfilt
from transformers import pipeline


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "audio" / "curated" / "zh"
META_CSV = ROOT / "assets" / "audio" / "curated" / "metadata.csv"
META_JSON = ROOT / "assets" / "audio" / "curated" / "metadata.json"
TARGET_SR = 44100

# ESD split IDs in this packaged dataset
EMOTION_ID = {"happy": "4", "sad": "6"}


def decode_audio(audio_bytes: bytes):
    wav, sr = sf.read(io.BytesIO(audio_bytes))
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav.astype(np.float32), sr


def quality_score(wav: np.ndarray, sr: int) -> float:
    # Prefer expressive and clean clips:
    # - higher energy/dynamics
    # - lower clipping and excessive high-frequency hiss
    rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-9))
    dyn = float(np.std(wav))
    clipping = float(np.mean(np.abs(wav) > 0.98))
    spec_flat = float(np.mean(librosa.feature.spectral_flatness(y=wav)))
    dur = min(len(wav) / sr, 8.0)
    return rms * 0.9 + dyn * 0.5 + dur * 0.06 - clipping * 3.0 - spec_flat * 0.35


def enhance_audio(wav: np.ndarray, sr: int) -> np.ndarray:
    # Trim obvious silence
    wav, _ = librosa.effects.trim(wav, top_db=35)

    # High-pass filter to reduce low-frequency rumble
    if len(wav) > 128:
        b, a = butter(N=2, Wn=60 / (sr / 2), btype="highpass")
        wav = filtfilt(b, a, wav).astype(np.float32)

    # Gentle compression for more stable loudness
    wav = np.tanh(1.25 * wav) / np.tanh(1.25)

    # Loudness normalization target RMS ~ -20 dBFS
    target_rms = 10 ** (-20 / 20)
    cur_rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-9))
    wav = wav * (target_rms / max(cur_rms, 1e-6))

    # Peak limit
    peak = float(np.max(np.abs(wav)) + 1e-9)
    if peak > 0.97:
        wav = wav * (0.97 / peak)
    return wav.astype(np.float32)


def pick_two_different_speakers():
    ds = load_dataset("yukat237/emotional-speech-audio-dataset-3eng-4noneng-updated", split="ESD")
    candidates = {"happy": [], "sad": []}

    for row in ds:
        for emo in ("happy", "sad"):
            if row["emotion"] != EMOTION_ID[emo]:
                continue
            wav, sr = decode_audio(row["audio"]["bytes"])
            score = quality_score(wav, sr)
            candidates[emo].append((score, row, wav, sr))

    for emo in candidates:
        candidates[emo].sort(key=lambda x: x[0], reverse=True)

    # Pick best happy first, then best sad with different speaker
    happy_best = candidates["happy"][0]
    happy_spk = happy_best[1]["speakerID"]
    sad_best = None
    for item in candidates["sad"]:
        if item[1]["speakerID"] != happy_spk:
            sad_best = item
            break
    if sad_best is None:
        raise RuntimeError("Cannot find sad sample with different speaker from happy.")
    return {"happy": happy_best, "sad": sad_best}


def transcribe_cn(asr_pipe, wav: np.ndarray, sr: int) -> str:
    wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    out = asr_pipe(
        {"array": wav16, "sampling_rate": 16000},
        generate_kwargs={"language": "chinese", "task": "transcribe"},
    )
    return out["text"].strip()


def update_metadata(entries):
    rows = list(csv.DictReader(META_CSV.open(encoding="utf-8")))
    for row in rows:
        if row["language"] == "zh" and row["emotion"] in entries:
            e = entries[row["emotion"]]
            row["transcript"] = e["transcript"]
            row["source_dataset"] = "yukat237/emotional-speech-audio-dataset-3eng-4noneng-updated (ESD split, refined quality)"
            row["source_ref"] = e["source_ref"]
            row["sample_rate"] = str(TARGET_SR)
    with META_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with META_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if item["language"] == "zh" and item["emotion"] in entries:
            e = entries[item["emotion"]]
            item["transcript"] = e["transcript"]
            item["source_dataset"] = "yukat237/emotional-speech-audio-dataset-3eng-4noneng-updated (ESD split, refined quality)"
            item["source_ref"] = e["source_ref"]
            item["sample_rate"] = TARGET_SR
    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    picks = pick_two_different_speakers()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")
    updates = {}

    for emo in ("happy", "sad"):
        _, row, wav, sr = picks[emo]
        wav = enhance_audio(wav, sr)
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
        out_path = OUT_DIR / f"zh_{emo}.wav"
        sf.write(out_path, wav, TARGET_SR, subtype="PCM_16")
        transcript = transcribe_cn(asr, wav, TARGET_SR)
        updates[emo] = {
            "source_ref": f"{row['audio']['path']}|speaker={row['speakerID']}",
            "transcript": transcript,
        }

    update_metadata(updates)
    print("Updated zh_happy.wav and zh_sad.wav with different speakers and enhanced quality.")
    print("happy source:", updates["happy"]["source_ref"])
    print("sad source:", updates["sad"]["source_ref"])


if __name__ == "__main__":
    main()
