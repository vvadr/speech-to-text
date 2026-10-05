"""Build and audit the Mendeley Karakalpak corpus from its paired v1 manifest.

Version 2 renamed CSV rows to kaa0... but kept the audio. Version 1 has the
original kaa1... filename-to-text mapping. Never infer a pair from row order.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import io
import json
import re
import shutil
import subprocess
import unicodedata
import wave
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "karakalpak-mendeley"
SOURCES = CORPUS / "sources"
V1_ARCHIVE = SOURCES / "DATASET_v1.7z"
V2_ARCHIVE = SOURCES / "DATASET_version2.7z"
RAW_AUDIO = SOURCES / "version2_extracted" / "DATASET"
ALIGNED = CORPUS / "aligned"
CLEANED = CORPUS / "cleaned"
MANIFEST = ALIGNED / "manifest.csv"
MEASUREMENTS = ALIGNED / "measurements.csv"
EXPECTED_SHA256 = {
    V1_ARCHIVE: "97ba2d6f457e3710cd305208aea1196b5c798a544e1b6c62409ae19f2b230918",
    V2_ARCHIVE: "03ddca8106f8e6247e556665381800eb458f58e2c4c6947bae0b99b290c6d91b",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def archive_crcs(path: Path) -> dict[str, int]:
    listing = subprocess.check_output(["7z", "l", "-slt", str(path)], text=True)
    result: dict[str, int] = {}
    name = ""
    for line in listing.splitlines():
        if line.startswith("Path = "):
            name = line.removeprefix("Path = ")
        elif line.startswith("CRC = ") and name.endswith(".wav"):
            result[Path(name).name] = int(line.removeprefix("CRC = "), 16)
    return result


def local_crc(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return value


def wav_info(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path), "rb") as audio:
        return (
            audio.getframerate(),
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getnframes() / audio.getframerate(),
        )


def source_container(path: Path) -> str:
    with path.open("rb") as handle:
        header = handle.read(12)
    if header.startswith(b"RIFF"):
        return "WAV"
    if b"ftyp" in header:
        return "MP4/AAC"
    raise ValueError(f"Unsupported audio container: {path}")


def align() -> None:
    for archive, expected in EXPECTED_SHA256.items():
        actual = digest(archive)
        if actual != expected:
            raise ValueError(f"Published SHA-256 mismatch: {archive}: {actual}")

    source_bytes = subprocess.check_output(["7z", "e", "-so", str(V1_ARCHIVE), "train.csv"])
    source_rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig"))))
    if not source_rows or set(source_rows[0]) != {"filename", "text"}:
        raise ValueError("Unexpected version 1 manifest schema")

    texts: dict[str, str] = {}
    counts: Counter[str] = Counter()
    conflicts: list[str] = []
    for row in source_rows:
        name, text = row["filename"], row["text"].strip()
        if not text:
            raise ValueError(f"Empty transcript: {name}")
        if name in texts and texts[name] != text:
            conflicts.append(name)
        texts[name] = text
        counts[name] += 1
    if conflicts:
        raise ValueError(f"Conflicting transcripts for {len(set(conflicts))} audio files")

    v1_crcs = archive_crcs(V1_ARCHIVE)
    v2_crcs = archive_crcs(V2_ARCHIVE)
    raw_files = {path.name: path for path in RAW_AUDIO.glob("*.wav")}
    if set(texts) != set(v1_crcs) or set(texts) != set(v2_crcs) or set(texts) != set(raw_files):
        raise ValueError("Version 1 transcript filenames and both audio sets differ")
    if v1_crcs != v2_crcs:
        raise ValueError("Version 1 and 2 audio CRCs differ")
    bad = [name for name, path in raw_files.items() if local_crc(path) != v1_crcs[name]]
    if bad:
        raise ValueError(f"Extracted audio differs from published archive: {bad[:5]}")

    rows = [
        {
            "audio_index": index,
            "filename": name,
            "sentence_latin": texts[name],
            "source_csv_row_count": counts[name],
            "source_container": source_container(raw_files[name]),
            "source_crc32": f"{v1_crcs[name]:08X}",
        }
        for index, name in enumerate(sorted(texts))
    ]
    ALIGNED.mkdir(parents=True, exist_ok=True)
    (ALIGNED / "publisher_train_v1.csv").write_bytes(source_bytes)
    write_csv(MANIFEST, list(rows[0]), rows)
    write_csv(
        ALIGNED / "source_pairs.csv", ["filename", "text"],
        [{"filename": row["filename"], "text": row["sentence_latin"]} for row in rows],
    )
    provenance = {
        "source": "https://data.mendeley.com/datasets/2th8jvft8f",
        "mapping_archive": str(V1_ARCHIVE.relative_to(ROOT)),
        "audio_archive": str(V2_ARCHIVE.relative_to(ROOT)),
        "sha256": {str(path.relative_to(ROOT)): value for path, value in EXPECTED_SHA256.items()},
        "publisher_rows": len(source_rows),
        "unique_audio_text_pairs": len(rows),
        "repeated_identical_rows": len(source_rows) - len(rows),
        "audio_crc_matches_between_versions": len(rows),
        "local_audio_crc_matches": len(rows),
        "source_containers": dict(Counter(row["source_container"] for row in rows)),
    }
    (ALIGNED / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Aligned {len(rows)} unique audio/transcript pairs from {len(source_rows)} publisher rows")


def ffmpeg_wav(source: Path, destination: Path, sample_rate: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.wav")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-threads", "1",
        "-y", "-i", str(source), "-ac", "1",
    ]
    if sample_rate is not None:
        command += ["-ar", str(sample_rate)]
    command += ["-c:a", "pcm_s16le", "-f", "wav", str(temporary)]
    subprocess.run(command, check=True, capture_output=True)
    temporary.replace(destination)


def prepare_one(row: dict[str, str]) -> None:
    name = row["filename"]
    source = RAW_AUDIO / name
    original = ALIGNED / "audio_original" / name
    converted = ALIGNED / "audio_16k" / name
    original.parent.mkdir(parents=True, exist_ok=True)
    if row["source_container"] == "WAV":
        if not original.is_file() or not filecmp.cmp(source, original, shallow=False):
            temporary = original.with_suffix(".partial.wav")
            shutil.copyfile(source, temporary)
            temporary.replace(original)
    elif not original.is_file() or not original.read_bytes().startswith(b"RIFF"):
        ffmpeg_wav(source, original)

    rate, channels, width, duration = wav_info(original)
    if channels not in (1, 2) or width != 2:
        raise ValueError(f"Unexpected decoded format: {name}")
    valid = False
    if converted.is_file():
        try:
            new_rate, new_channels, new_width, new_duration = wav_info(converted)
            valid = (
                (new_rate, new_channels, new_width) == (16_000, 1, 2)
                and abs(duration - new_duration) < 0.002
            )
        except (OSError, wave.Error):
            pass
    if not valid:
        if rate == 16_000 and channels == 1:
            temporary = converted.with_suffix(".partial.wav")
            shutil.copyfile(original, temporary)
            temporary.replace(converted)
        else:
            ffmpeg_wav(original, converted, 16_000)
    if abs(duration - wav_info(converted)[3]) >= 0.002:
        raise ValueError(f"Resampled duration changed: {name}")


def prepare_audio(workers: int) -> None:
    rows = read_csv(MANIFEST)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(prepare_one, rows))
    expected = {row["filename"] for row in rows}
    for folder in ("audio_original", "audio_16k"):
        actual = {path.name for path in (ALIGNED / folder).glob("*.wav")}
        if actual != expected:
            raise ValueError(f"{folder}: audio file set differs from manifest")
    print(f"Prepared {len(rows)} original/decoded and 16 kHz WAV pairs")


def estimate_snr(path: Path) -> float | None:
    import librosa
    import numpy as np

    audio, rate = librosa.load(path, sr=None)
    intervals = librosa.effects.split(
        audio, top_db=30, frame_length=round(0.025 * rate), hop_length=round(0.010 * rate)
    )
    active = np.zeros(len(audio), dtype=bool)
    for start, end in intervals:
        active[start:end] = True
    if not active.any() or active.all():
        return None
    speech_power = np.mean(audio[active] ** 2)
    quiet_power = np.mean(audio[~active] ** 2)
    return round(float(10 * np.log10((speech_power + 1e-12) / (quiet_power + 1e-12))), 6)


def features() -> None:
    rows = read_csv(MANIFEST)
    output = []
    for index, row in enumerate(rows):
        name = row["filename"]
        original = ALIGNED / "audio_original" / name
        converted = ALIGNED / "audio_16k" / name
        original_info = wav_info(original)
        converted_info = wav_info(converted)
        if original_info[1] not in (1, 2) or original_info[2] != 2 or converted_info[:3] != (16_000, 1, 2):
            raise ValueError(f"Invalid WAV format: {name}")
        if abs(original_info[3] - converted_info[3]) >= 0.002:
            raise ValueError(f"Duration mismatch: {name}")
        output.append({
            "audio_index": index,
            "filename": name,
            "sentence_latin": row["sentence_latin"],
            "duration": converted_info[3],
            "snr_original": estimate_snr(original),
            "snr_16k": estimate_snr(converted),
            "source_csv_row_count": row["source_csv_row_count"],
            "source_container": row["source_container"],
        })
        if (index + 1) % 100 == 0:
            print(f"Measured {index + 1}/{len(rows)} recordings", flush=True)
    write_csv(MEASUREMENTS, list(output[0]), output)
    print(f"Wrote measurements for {len(output)} aligned pairs")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join("".join(c if c.isalpha() or c.isdigit() or c.isspace() else " " for c in text).lower().split())


def reviewed_expansions() -> dict[str, list[dict[str, str]]]:
    path = ALIGNED / "abbreviation_review.csv"
    if not path.is_file():
        return {}
    expansions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        if row.get("verified", "").strip().lower() not in {"yes", "true", "1"}:
            continue
        name = row["source_filename"]
        abbreviation = normalize(row["abbreviation"])
        spoken = normalize(row["spoken_form"])
        if not abbreviation or not spoken or abbreviation == spoken or not row.get("evidence", "").strip():
            raise ValueError(f"Incomplete verified abbreviation decision for {name}")
        expansions[name].append({"abbreviation": abbreviation, "spoken_form": spoken})
    return expansions


def normalized_transcript(text: str, name: str, expansions: dict[str, list[dict[str, str]]]) -> tuple[str, int]:
    result = normalize(text)
    applied = 0
    for decision in expansions.get(name, []):
        abbreviation = decision["abbreviation"]
        pattern = rf"(?<!\w){re.escape(abbreviation)}(?!\w)"
        result, count = re.subn(pattern, decision["spoken_form"], result)
        if count != 1:
            raise ValueError(f"Expected one {abbreviation!r} in {name}; found {count}")
        applied += 1
    return result, applied


def write_layout_guides(pair_count: int) -> None:
    guides = {
        CORPUS / "README.md": (
            "# Karakalpak Mendeley dataset\n\n"
            f"Use `cleaned/train.csv` with the {pair_count:,} matching 16 kHz WAVs in `cleaned/audio/`.\n\n"
            "`cleaned/` is the final dataset. `aligned/` holds intermediate pairs and measurements. "
            "`sources/` holds the original Mendeley files.\n"
        ),
        CLEANED / "README.md": (
            "# Cleaned Mendeley dataset\n\n"
            "Use `train.csv` and `audio/` together. Each `filename` in the CSV names one WAV in `audio/`.\n\n"
            f"There are {pair_count:,} paired rows and WAVs. `metadata.csv` adds source traceability; "
            "`audio_original/` has the matching pre-resampling recordings.\n"
        ),
    }
    for path, content in guides.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8")


def clean() -> None:
    rows = read_csv(MEASUREMENTS)
    kept = []
    for row in rows:
        word_count = len(row["sentence_latin"].split())
        if float(row["duration"]) < 2 or word_count < 4:
            continue
        index = len(kept)
        kept.append({
            "audio_index": index,
            "filename": f"audio_{index:06d}.wav",
            "source_filename": row["filename"],
            "sentence_latin": row["sentence_latin"],
            "duration": row["duration"],
            "snr_original": row["snr_original"],
            "snr_16k": row["snr_16k"],
            "word_count": word_count,
            "source_csv_row_count": row["source_csv_row_count"],
            "source_container": row["source_container"],
        })
    write_csv(ALIGNED / "selection.csv", list(kept[0]), kept)
    for source_folder, target_folder in (("audio_original", "audio_original"), ("audio_16k", "audio")):
        target = CLEANED / target_folder
        target.mkdir(parents=True, exist_ok=True)
        expected = {row["filename"] for row in kept}
        for row in kept:
            source = ALIGNED / source_folder / row["source_filename"]
            destination = target / row["filename"]
            if not destination.is_file() or not filecmp.cmp(source, destination, shallow=False):
                temporary = destination.with_suffix(".partial.wav")
                shutil.copyfile(source, temporary)
                temporary.replace(destination)
        for stale in target.glob("*.wav"):
            if stale.name not in expected:
                stale.unlink()

    expansions = reviewed_expansions()
    normalized = []
    applied = 0
    for row in kept:
        cleaned_text, count = normalized_transcript(
            row["sentence_latin"], row["source_filename"], expansions
        )
        applied += count
        normalized.append({
            **row,
            "source_sentence_latin": row["sentence_latin"],
            "sentence_latin": cleaned_text,
            "normalized_word_count": len(cleaned_text.split()),
        })
    write_csv(CLEANED / "metadata.csv", list(normalized[0]), normalized)
    write_csv(
        CLEANED / "train.csv", ["filename", "text"],
        [{"filename": row["filename"], "text": row["sentence_latin"]} for row in normalized],
    )
    write_layout_guides(len(kept))
    print(f"Kept {len(kept)} paired recordings; applied {applied} verified abbreviation expansions")


def audit() -> None:
    manifest = read_csv(MANIFEST)
    data = read_csv(MEASUREMENTS)
    cleaned = read_csv(ALIGNED / "selection.csv")
    normalized = read_csv(CLEANED / "metadata.csv")
    source_train = read_csv(ALIGNED / "source_pairs.csv")
    clean_train = read_csv(CLEANED / "train.csv")
    if len(manifest) != 2022 or len(data) != 2022 or len(cleaned) != len(normalized):
        raise ValueError("Output row count mismatch")
    if source_train != [
        {"filename": row["filename"], "text": row["sentence_latin"]} for row in manifest
    ] or clean_train != [
        {"filename": row["filename"], "text": row["sentence_latin"]} for row in normalized
    ]:
        raise ValueError("Train CSV rows do not match their paired audio manifests")
    if [row["filename"] for row in manifest] != [row["filename"] for row in data]:
        raise ValueError("Measured rows lost source filename order")
    expected_retained = [
        row["filename"] for row in data
        if float(row["duration"]) >= 2 and len(row["sentence_latin"].split()) >= 4
    ]
    if [row["source_filename"] for row in cleaned] != expected_retained:
        raise ValueError("Cleaned rows differ from duration and word-count filter")
    expansions = reviewed_expansions()
    for index, (row, normalized_row) in enumerate(zip(cleaned, normalized)):
        if (
            row["audio_index"] != str(index)
            or row["filename"] != f"audio_{index:06d}.wav"
            or row["filename"] != normalized_row["filename"]
            or row["source_filename"] != normalized_row["source_filename"]
        ):
            raise ValueError(f"Cleaned audio/text order differs at {index}")
        expected_text, _ = normalized_transcript(
            row["sentence_latin"], row["source_filename"], expansions
        )
        if (
            normalized_row["source_sentence_latin"] != row["sentence_latin"]
            or normalized_row["sentence_latin"] != expected_text
            or normalized_row["normalized_word_count"] != str(len(expected_text.split()))
        ):
            raise ValueError(f"Cleaned transcript differs from reviewed text at {index}")
    for source_folder, target_folder in (("audio_original", "audio_original"), ("audio_16k", "audio")):
        expected = {row["filename"] for row in cleaned}
        actual = {path.name for path in (CLEANED / target_folder).glob("*.wav")}
        if actual != expected:
            raise ValueError(f"Cleaned/{target_folder} has missing or extra audio files")
        for row in cleaned:
            source = ALIGNED / source_folder / row["source_filename"]
            destination = CLEANED / target_folder / row["filename"]
            if not filecmp.cmp(source, destination, shallow=False):
                raise ValueError(f"Cleaned audio differs from source: {destination}")
    print(f"Audit passed: {len(data)} aligned source pairs, {len(cleaned)} cleaned pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "align", "audio", "features", "clean", "audit"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    stages = ("align", "audio", "features", "clean", "audit") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"Stage: {stage}", flush=True)
        {
            "align": align,
            "audio": lambda: prepare_audio(args.workers),
            "features": features,
            "clean": clean,
            "audit": audit,
        }[stage]()


if __name__ == "__main__":
    main()
