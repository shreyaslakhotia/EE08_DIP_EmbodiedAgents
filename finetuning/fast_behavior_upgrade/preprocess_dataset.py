#!/usr/bin/env python3
"""
Fast weak-supervision preprocessing pipeline for study-scenario behavioral fine-tuning.

What it does:
1) Recursively scans dataset root for videos/audio/images.
2) Uses scenario folder names as weak labels.
3) Extracts 2-4 frames per video (configurable).
4) Optionally transcribes audio with faster-whisper / whisper.
5) Builds synthetic multimodal instruction-response samples in JSONL.
6) Writes train/val JSONL split with robust error logging.

This script is designed for messy, real-world datasets and does not crash on bad files.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

SCENARIO_MAP = {
    "Doomscrolling": {"affect": "distracted / avoidant", "engagement": "off_task", "style": "gentle_redirect"},
    "Actually Studying!!!!!": {"affect": "focused", "engagement": "on_task", "style": "encourage_or_silent"},
    "Fidgeting": {"affect": "restless / stressed", "engagement": "low_focus", "style": "grounding_or_break"},
    "Playing game": {"affect": "engaged_elsewhere", "engagement": "off_task", "style": "soft_redirect"},
    "Stretching": {"affect": "neutral / recovering", "engagement": "break", "style": "positive_reinforcement"},
    "Looking through lecture notes": {"affect": "engaged", "engagement": "on_task", "style": "light_encouragement"},
    "Looking at quiz or exam result": {"affect": "anxious / reflective", "engagement": "uncertain", "style": "reassurance"},
    "Watching lecture videos": {"affect": "passive_learning", "engagement": "semi_on_task", "style": "encourage_active_learning"},
    "Setting up timetable": {"affect": "planning", "engagement": "on_task", "style": "supportive"},
    "Empty table": {"affect": "no_person", "engagement": "none", "style": "no_intervention"},
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_") or "item"


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def find_scenario_from_path(path: Path) -> Optional[str]:
    norm_to_canonical = {normalize_key(k): k for k in SCENARIO_MAP}
    for part in reversed(path.parts):
        key = normalize_key(part)
        if key in norm_to_canonical:
            return norm_to_canonical[key]
    return None


def style_to_response(style: str, scenario: str, transcript: str) -> str:
    has_transcript = bool(transcript.strip())

    if style == "gentle_redirect":
        return (
            "It looks like you are drifting into doomscrolling. No stress, this happens to everyone. "
            "How about a 10-minute focused sprint on one small task, then you can decide what to do next?"
        )
    if style == "encourage_or_silent":
        return (
            "You look focused and in study mode. Keep this momentum going. "
            "I can stay quiet or help with a quick summary when you want."
        )
    if style == "grounding_or_break":
        return (
            "You seem restless right now. Let us reset with a short breathing break, then pick one tiny next step to restart focus."
        )
    if style == "soft_redirect":
        return (
            "Looks like your attention is on gaming right now. If your goal is to study, "
            "let us park the game for 15 minutes and finish one quick revision goal first."
        )
    if style == "positive_reinforcement":
        return (
            "Nice reset. Stretching can help your brain recover. "
            "When you are ready, we can continue with one manageable study task."
        )
    if style == "light_encouragement":
        return (
            "Great, you are reviewing lecture notes. That is solid progress. "
            "Want a quick recap checklist so you can confirm key points fast?"
        )
    if style == "reassurance":
        return (
            "You may be processing a result right now. It is okay to feel anxious or reflective. "
            "Let us focus on one constructive next action you can control today."
        )
    if style == "encourage_active_learning":
        return (
            "Watching lecture videos is useful. To lock it in, pause every few minutes and summarize one key idea in your own words."
        )
    if style == "supportive":
        return (
            "Nice planning behavior. Building a timetable is a strong study habit. "
            "Let us keep it realistic with short focused blocks and clear breaks."
        )
    if style == "no_intervention":
        return "No intervention needed right now."

    base = "I am here with you. Let us take one small next step together."
    if has_transcript:
        base += " I also used the available transcript context to guide this suggestion."
    return base


def build_user_prompt(scenario: str, transcript: str) -> str:
    transcript_text = transcript.strip() if transcript.strip() else "N/A"
    return (
        "You are an emotionally intelligent study buddy robot.\n\n"
        "A student is in the following situation:\n"
        f"{scenario}\n\n"
        "Transcript (if any):\n"
        f"{transcript_text}\n\n"
        "Describe:\n"
        "1. What the student is doing\n"
        "2. Their likely state\n"
        "3. Whether intervention is needed\n"
        "4. Provide a short empathetic response"
    )


def build_target_text(scenario: str, affect: str, engagement: str, style: str, transcript: str) -> str:
    response = style_to_response(style=style, scenario=scenario, transcript=transcript)
    return (
        f"Activity: {scenario}\n"
        f"State: {affect}\n"
        f"Engagement: {engagement}\n"
        f"Intervention: {style}\n\n"
        f"Response: {response}"
    )


def check_ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def transcribe_with_faster_whisper(audio_path: Path, model_name: str) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    return text.strip()


def transcribe_with_openai_whisper(audio_path: Path, model_name: str) -> str:
    import whisper

    model = whisper.load_model(model_name, device="cpu")
    result = model.transcribe(str(audio_path), fp16=False)
    return (result.get("text") or "").strip()


@dataclass
class PipelineStats:
    total_files_seen: int = 0
    total_media_matched: int = 0
    unknown_scenario_skipped: int = 0
    bad_files_skipped: int = 0
    video_count: int = 0
    audio_count: int = 0
    image_count: int = 0
    transcript_success: int = 0
    transcript_fail: int = 0
    samples_created: int = 0


def write_error(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def extract_frames(video_path: Path, out_dir: Path, frames_per_video: int, error_log: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: List[Path] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        write_error(error_log, f"[VIDEO_OPEN_FAIL] {video_path}")
        return frame_paths

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        write_error(error_log, f"[VIDEO_NO_FRAMES] {video_path}")
        cap.release()
        return frame_paths

    indices = sorted({int((i + 1) * frame_count / (frames_per_video + 1)) for i in range(frames_per_video)})

    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx, 0))
        ok, frame = cap.read()
        if not ok or frame is None:
            write_error(error_log, f"[FRAME_READ_FAIL] {video_path} frame={frame_idx}")
            continue

        out_name = f"{safe_slug(video_path.stem)}_f{idx:02d}.jpg"
        out_path = out_dir / out_name
        ok_write = cv2.imwrite(str(out_path), frame)
        if not ok_write:
            write_error(error_log, f"[FRAME_WRITE_FAIL] {out_path}")
            continue
        frame_paths.append(out_path)

    cap.release()
    return frame_paths


def extract_audio_from_video(video_path: Path, wav_out_path: Path, error_log: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_out_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as exc:
        write_error(error_log, f"[AUDIO_EXTRACT_FAIL] {video_path} :: {exc}")
        return False


def try_transcribe(
    audio_path: Path,
    enable: bool,
    backend: str,
    whisper_model: str,
    error_log: Path,
) -> Tuple[str, bool]:
    if not enable:
        return "", False

    try:
        if backend == "faster-whisper":
            return transcribe_with_faster_whisper(audio_path, whisper_model), True
        if backend == "whisper":
            return transcribe_with_openai_whisper(audio_path, whisper_model), True
        write_error(error_log, f"[TRANSCRIBE_BACKEND_UNKNOWN] backend={backend}")
        return "", False
    except Exception as exc:
        write_error(error_log, f"[TRANSCRIBE_FAIL] {audio_path} :: {exc}")
        return "", False


def make_record(image_path: Path, scenario: str, transcript: str) -> dict:
    label = SCENARIO_MAP[scenario]
    user_text = build_user_prompt(scenario=scenario, transcript=transcript)
    assistant_text = build_target_text(
        scenario=scenario,
        affect=label["affect"],
        engagement=label["engagement"],
        style=label["style"],
        transcript=transcript,
    )
    return {
        "image": str(image_path),
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def split_train_val(samples: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if not samples:
        return [], []

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    n_val = int(len(shuffled) * val_ratio)
    n_val = max(1, n_val) if len(shuffled) > 1 else 0
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def iter_media_files(data_root: Path):
    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS or ext in AUDIO_EXTS or ext in IMAGE_EXTS:
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess weak-labeled multimodal dataset into JSONL for Qwen2.5-VL LoRA fine-tuning")
    parser.add_argument("--data_root", type=Path, required=True, help="Root dataset folder (recursive scan)")
    parser.add_argument("--output_path", type=Path, required=True, help="Output JSONL path for all samples")
    parser.add_argument("--max_files", type=int, default=0, help="Optional debug limit for matched media files (0 = no limit)")
    parser.add_argument("--frames_per_video", type=int, default=3, help="Frames to extract per video (recommended 2-4)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--enable_transcription", action="store_true", help="If set, run Whisper transcription when audio exists")
    parser.add_argument(
        "--transcriber",
        type=str,
        default="faster-whisper",
        choices=["faster-whisper", "whisper"],
        help="Whisper backend",
    )
    parser.add_argument("--whisper_model", type=str, default="small", help="Whisper model size/name")
    args = parser.parse_args()

    if args.frames_per_video < 1:
        print("[ERROR] --frames_per_video must be >= 1")
        return 1

    if not args.data_root.exists():
        print(f"[ERROR] data_root not found: {args.data_root}")
        return 1

    output_all = args.output_path
    output_all.parent.mkdir(parents=True, exist_ok=True)
    output_train = output_all.parent / "train.jsonl"
    output_val = output_all.parent / "val.jsonl"
    error_log = output_all.parent / "preprocess_errors.log"
    if error_log.exists():
        error_log.unlink()

    frames_root = output_all.parent / "extracted_frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    ffmpeg_ok = check_ffmpeg_available()
    if args.enable_transcription and not ffmpeg_ok:
        print("[WARN] ffmpeg not found. Video-audio transcription will be skipped unless files are audio-only.")

    media_files = list(iter_media_files(args.data_root))
    total_media = len(media_files)
    if args.max_files > 0:
        media_files = media_files[: args.max_files]

    stats = PipelineStats(total_files_seen=total_media, total_media_matched=len(media_files))
    samples: List[dict] = []

    print("\n=== PREPROCESS START ===")
    print(f"data_root: {args.data_root}")
    print(f"output_all: {output_all}")
    print(f"frames_per_video: {args.frames_per_video}")
    print(f"matched media files: {len(media_files)} (before debug limit: {total_media})")
    print(f"transcription enabled: {args.enable_transcription} ({args.transcriber}, model={args.whisper_model})")

    for media_path in tqdm(media_files, desc="Processing media"):
        scenario = find_scenario_from_path(media_path)
        if scenario is None:
            stats.unknown_scenario_skipped += 1
            write_error(error_log, f"[UNKNOWN_SCENARIO] {media_path}")
            continue

        ext = media_path.suffix.lower()
        try:
            if ext in VIDEO_EXTS:
                stats.video_count += 1
                rel = media_path.relative_to(args.data_root)
                frame_dir = frames_root / rel.parent / safe_slug(media_path.stem)
                frame_paths = extract_frames(
                    video_path=media_path,
                    out_dir=frame_dir,
                    frames_per_video=args.frames_per_video,
                    error_log=error_log,
                )
                if not frame_paths:
                    stats.bad_files_skipped += 1
                    continue

                transcript = ""
                if args.enable_transcription and ffmpeg_ok:
                    with tempfile.TemporaryDirectory(prefix="audio_extract_") as td:
                        wav_path = Path(td) / "tmp_audio.wav"
                        if extract_audio_from_video(media_path, wav_path, error_log):
                            transcript, ok = try_transcribe(
                                audio_path=wav_path,
                                enable=True,
                                backend=args.transcriber,
                                whisper_model=args.whisper_model,
                                error_log=error_log,
                            )
                            if ok and transcript:
                                stats.transcript_success += 1
                            elif ok:
                                stats.transcript_fail += 1

                for fp in frame_paths:
                    samples.append(make_record(image_path=fp, scenario=scenario, transcript=transcript))
                    stats.samples_created += 1

            elif ext in AUDIO_EXTS:
                stats.audio_count += 1
                transcript = ""
                if args.enable_transcription:
                    transcript, ok = try_transcribe(
                        audio_path=media_path,
                        enable=True,
                        backend=args.transcriber,
                        whisper_model=args.whisper_model,
                        error_log=error_log,
                    )
                    if ok and transcript:
                        stats.transcript_success += 1
                    elif ok:
                        stats.transcript_fail += 1

                # Audio-only fallback: attach transcript to a representative frame.
                # If no frame exists yet for this scenario, skip to keep strict image+text format.
                scenario_frames = [s for s in samples if s["messages"][1]["content"].startswith(f"Activity: {scenario}")]
                if scenario_frames:
                    rep_image = Path(scenario_frames[0]["image"])
                    samples.append(make_record(image_path=rep_image, scenario=scenario, transcript=transcript))
                    stats.samples_created += 1
                else:
                    write_error(error_log, f"[AUDIO_ONLY_NO_FRAME_CONTEXT] {media_path}")
                    stats.bad_files_skipped += 1

            elif ext in IMAGE_EXTS:
                stats.image_count += 1
                samples.append(make_record(image_path=media_path, scenario=scenario, transcript=""))
                stats.samples_created += 1

        except Exception as exc:
            stats.bad_files_skipped += 1
            write_error(error_log, f"[UNHANDLED_FILE_ERROR] {media_path} :: {exc}")

    train_samples, val_samples = split_train_val(samples, val_ratio=args.val_ratio, seed=args.seed)

    with output_all.open("w", encoding="utf-8") as f_all:
        for item in samples:
            f_all.write(json.dumps(item, ensure_ascii=False) + "\n")

    with output_train.open("w", encoding="utf-8") as f_train:
        for item in train_samples:
            f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

    with output_val.open("w", encoding="utf-8") as f_val:
        for item in val_samples:
            f_val.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n=== PREPROCESS SUMMARY ===")
    print(f"total files discovered (media types only): {stats.total_files_seen}")
    print(f"files processed after max_files limit: {stats.total_media_matched}")
    print(f"video files: {stats.video_count}")
    print(f"audio files: {stats.audio_count}")
    print(f"image files: {stats.image_count}")
    print(f"unknown scenario skipped: {stats.unknown_scenario_skipped}")
    print(f"bad files skipped: {stats.bad_files_skipped}")
    print(f"transcript success: {stats.transcript_success}")
    print(f"transcript fail/empty: {stats.transcript_fail}")
    print(f"samples created: {stats.samples_created}")
    print(f"all samples jsonl: {output_all}")
    print(f"train split jsonl: {output_train}")
    print(f"val split jsonl: {output_val}")
    print(f"error log: {error_log}")

    if not samples:
        print("[ERROR] No samples generated. Check scenario folder names and data quality.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
