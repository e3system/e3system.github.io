import csv
import io
import json
import re
import shutil
import tarfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import HfFileSystem
from transformers import pipeline


TARGET_SR = 44100
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "audio" / "curated"
META_JSON = OUT_DIR / "metadata.json"
META_CSV = OUT_DIR / "metadata.csv"


EN_TARGET_EMOTIONS = ["happy", "sad", "angry", "fear", "neutral"]

ESD_EMOTION_ID = {
    # ESD merged split stores numeric IDs.
    # Based on sentence ID blocks in this packaged version:
    # 1-350 neutral, 351-700 angry, 701-1050 happy, 1051-1400 sad, 1401-1750 surprise.
    "happy": "4",
    "sad": "6",
}


def reset_output_dir() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for lang in ["zh", "ja", "en"]:
        (OUT_DIR / lang).mkdir(parents=True, exist_ok=True)


def decode_audio_bytes(audio_bytes: bytes):
    wav, sr = sf.read(io.BytesIO(audio_bytes))
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav.astype(np.float32), sr


def resample_to_44k1(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return wav
    return librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)


def save_wav(path: Path, wav: np.ndarray) -> None:
    sf.write(path, wav, TARGET_SR, subtype="PCM_16")


def waveform_score(wav: np.ndarray, sr: int) -> float:
    abs_wav = np.abs(wav)
    rms = float(np.mean(abs_wav))
    dyn = float(np.std(abs_wav))
    dur = min(len(wav) / sr, 8.0)
    return rms * 0.6 + dyn * 0.3 + dur * 0.1


def transcribe(asr_pipe, wav: np.ndarray, sr: int, lang: str) -> str:
    wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    result = asr_pipe(
        {"array": wav16, "sampling_rate": 16000},
        generate_kwargs={"language": lang, "task": "transcribe"},
    )
    return result["text"].strip()


def pick_chinese_from_esd(asr_pipe):
    ds = load_dataset("yukat237/emotional-speech-audio-dataset-3eng-4noneng-updated", split="ESD")
    chosen = {}
    used_sentence = set()

    for emotion, eid in [("happy", ESD_EMOTION_ID["happy"]), ("sad", ESD_EMOTION_ID["sad"])]:
        candidates = []
        for row in ds:
            if row["emotion"] != eid:
                continue
            sid = row["sentenceID"]
            if sid in used_sentence:
                continue
            wav, sr = decode_audio_bytes(row["audio"]["bytes"])
            score = waveform_score(wav, sr)
            candidates.append((score, sid, row, wav, sr))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates:
            raise RuntimeError(f"No Chinese ESD candidate found for {emotion}.")
        best = candidates[0]
        used_sentence.add(best[1])
        transcript = transcribe(asr_pipe, best[3], best[4], "chinese")
        chosen[emotion] = {
            "row": best[2],
            "wav": best[3],
            "sr": best[4],
            "transcript": transcript,
        }
    return chosen


def pick_japanese_from_jvnv(asr_pipe):
    # User asked JKNV; using widely available JVNV corpus package.
    ds = load_dataset("yukat237/emotional-speech-audio-dataset-3eng-4noneng-updated", split="JVNV")
    token_map = {
        "happy": "japanese_happy__",
        "sad": "japanese_sad__",
        "angry": "japanese_angry__",
    }
    chosen = {}
    used_sid = set()
    for emotion, token in token_map.items():
        candidates = []
        for row in ds:
            p = (row["audio"]["path"] or "").lower()
            if token not in p:
                continue
            sid = row["sentenceID"]
            if sid in used_sid:
                continue
            wav, sr = decode_audio_bytes(row["audio"]["bytes"])
            score = waveform_score(wav, sr)
            candidates.append((score, sid, row, wav, sr))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates:
            raise RuntimeError(f"No Japanese JVNV candidate found for {emotion}.")
        best = candidates[0]
        used_sid.add(best[1])
        transcript = transcribe(asr_pipe, best[3], best[4], "japanese")
        chosen[emotion] = {
            "row": best[2],
            "wav": best[3],
            "sr": best[4],
            "transcript": transcript,
        }
    return chosen


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def emotion_keyword_score(sentence_lower: str, emotion: str) -> float:
    kws = {
        "happy": ["happy", "excited", "awesome", "great", "amazing", "love", "wonderful", "glad"],
        "sad": ["sad", "sorry", "unfortunately", "depressed", "cry", "lonely", "heartbroken", "grief"],
        "angry": ["angry", "mad", "furious", "hate", "damn", "annoyed", "rage", "upset"],
        "fear": ["fear", "afraid", "scared", "terrified", "panic", "anxious", "frightened"],
        "neutral": ["today", "first", "next", "then", "also", "because", "actually"],
    }
    hits = sum(1 for w in kws[emotion] if w in sentence_lower)
    punct = sentence_lower.count("!") * (0.7 if emotion in {"happy", "angry", "fear"} else 0.2)
    return hits * 2.0 + punct


def crop_one_sentence(wav: np.ndarray, sr: int, text: str, sentence: str) -> np.ndarray:
    full = text.strip()
    if not full:
        clip_len = int(min(len(wav), sr * 4.0))
        return wav[:clip_len]
    start_char = max(full.find(sentence), 0)
    end_char = start_char + len(sentence)
    total_chars = max(len(full), 1)
    start_ratio = start_char / total_chars
    end_ratio = min(end_char / total_chars, 1.0)
    if end_ratio - start_ratio < 0.18:
        end_ratio = min(start_ratio + 0.25, 1.0)
    s = int(start_ratio * len(wav))
    e = int(end_ratio * len(wav))
    e = min(max(e, s + int(sr * 1.2)), len(wav))
    return wav[s:e]


def stream_emilia_english_candidates(max_shards: int = 1):
    # Use authenticated streaming to avoid downloading full tar shards to disk.
    shard_names = ["Emilia-YODAS/EN/EN-B001361.tar"][:max_shards]
    fs = HfFileSystem()
    for shard in shard_names:
        remote_path = f"datasets/amphion/Emilia-Dataset/{shard}"
        stream = fs.open(remote_path, "rb")
        tf = tarfile.open(fileobj=stream, mode="r|")
        pending_meta = None
        for member in tf:
            if not member.isfile():
                continue
            if member.name.endswith(".json"):
                pending_meta = json.load(tf.extractfile(member))
            elif member.name.endswith(".mp3") and pending_meta is not None:
                mp3_bytes = tf.extractfile(member).read()
                yield pending_meta, mp3_bytes, member.name
                pending_meta = None


def pick_english_from_emilia():
    chosen = {}
    used_transcript = set()
    for meta, mp3_bytes, member_name in stream_emilia_english_candidates(max_shards=1):
        text = str(meta.get("text", "")).strip()
        sentences = split_sentences(text)
        if not sentences:
            continue
        try:
            wav, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None, mono=True)
        except Exception:
            continue
        if len(wav) < sr * 1.2:
            continue
        wav = wav.astype(np.float32)
        base_score = waveform_score(wav, sr)

        for emotion in EN_TARGET_EMOTIONS:
            if emotion in chosen:
                continue
            best_sentence = None
            best_score = -1e9
            for sentence in sentences[:4]:
                key_score = emotion_keyword_score(sentence.lower(), emotion)
                total_score = key_score + base_score
                if total_score > best_score:
                    best_score = total_score
                    best_sentence = sentence
            if best_sentence is None:
                continue
            # Emotion gating to avoid random neutral assignment.
            if emotion != "neutral" and emotion_keyword_score(best_sentence.lower(), emotion) < 1.0:
                continue
            if emotion == "neutral" and "!" in best_sentence:
                continue
            transcript = best_sentence
            if transcript in used_transcript:
                continue
            cropped = crop_one_sentence(wav, sr, text, best_sentence)
            chosen[emotion] = {
                "wav": cropped,
                "sr": sr,
                "transcript": transcript,
                "source_ref": member_name,
            }
            used_transcript.add(transcript)
        if len(chosen) == len(EN_TARGET_EMOTIONS):
            break

    missing = [e for e in EN_TARGET_EMOTIONS if e not in chosen]
    if missing:
        raise RuntimeError(f"Emilia candidates missing emotions: {missing}. Increase scanned shards.")
    return chosen


def main():
    reset_output_dir()
    metadata = []
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")

    zh = pick_chinese_from_esd(asr_pipe)
    for emotion in ["happy", "sad"]:
        item = zh[emotion]
        out = OUT_DIR / "zh" / f"zh_{emotion}.wav"
        save_wav(out, resample_to_44k1(item["wav"], item["sr"]))
        metadata.append(
            {
                "language": "zh",
                "emotion": emotion,
                "path": str(out.relative_to(ROOT)),
                "transcript": item["transcript"],
                "source_dataset": "yukat237/emotional-speech-audio-dataset-3eng-4noneng-updated (ESD split)",
                "source_ref": item["row"]["audio"]["path"],
                "sample_rate": TARGET_SR,
            }
        )

    ja = pick_japanese_from_jvnv(asr_pipe)
    for emotion in ["happy", "sad", "angry"]:
        item = ja[emotion]
        out = OUT_DIR / "ja" / f"ja_{emotion}.wav"
        save_wav(out, resample_to_44k1(item["wav"], item["sr"]))
        metadata.append(
            {
                "language": "ja",
                "emotion": emotion,
                "path": str(out.relative_to(ROOT)),
                "transcript": item["transcript"],
                "source_dataset": "yukat237/emotional-speech-audio-dataset-3eng-4noneng-updated (JVNV split; per request JKNV)",
                "source_ref": item["row"]["audio"]["path"],
                "sample_rate": TARGET_SR,
            }
        )

    en = pick_english_from_emilia()
    for emotion in EN_TARGET_EMOTIONS:
        item = en[emotion]
        out = OUT_DIR / "en" / f"en_{emotion}.wav"
        save_wav(out, resample_to_44k1(item["wav"], item["sr"]))
        metadata.append(
            {
                "language": "en",
                "emotion": emotion,
                "path": str(out.relative_to(ROOT)),
                "transcript": item["transcript"],
                "source_dataset": "amphion/Emilia-Dataset (EN shard stream)",
                "source_ref": item["source_ref"],
                "sample_rate": TARGET_SR,
            }
        )

    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with META_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "language",
                "emotion",
                "path",
                "transcript",
                "source_dataset",
                "source_ref",
                "sample_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Generated {len(metadata)} files.")
    print(f"Metadata written to: {META_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
