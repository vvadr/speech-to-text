"""Organize local ASR corpora, preserving source pairs and shared train/test assignments.

No network access or transcript inference. Build in a staging folder, audit all
outputs, then publish. --cleanup removes only verified redundant legacy files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPORA = {"hf": "karakalpak-audio-dataset-hf", "mendeley": "karakalpak-mendeley"}
FIELDS = ["sample_id", "filename", "audio_path", "original_audio_path", "source",
          "source_filename", "source_split", "text", "sentence_latin",
          "source_sentence_latin", "sentence_cyrillic", "duration", "word_count",
          "snr_original", "snr_16k", "audio_sha256", "split"]
TRAIN_FIELDS = ["audio_path", "text", "duration"]
METADATA_FIELDS = [*TRAIN_FIELDS, "split"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audio_info(path):
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"Expected mono 16 kHz PCM16: {path}")
        frames = audio.readframes(audio.getnframes())
        if len(frames) != audio.getnframes() * 2 or not frames:
            raise ValueError(f"Truncated or empty audio: {path}")
        return hashlib.sha256(frames).hexdigest(), audio.getnframes() / 16000


def normalized_key(text):
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).split())


def load_inputs():
    records = []
    for source, name in CORPORA.items():
        root = DATA / name
        if source == "hf":
            raw = root / "sources" if (root / "sources/data.csv").exists() else root
            original_splits = {row["filename"]: split for split in ("train", "test")
                               for row in read_csv(raw / f"{split}.csv")}
            clean = read_csv(root / "cleaned/data.csv")
            normalized = read_csv(root / "cleaned/removed_annotations.csv")
            if len(clean) != len(normalized):
                raise ValueError("HF normalized and selected row counts differ")
            originals = {row["filename"]: row for row in read_csv(raw / "data.csv")}
        else:
            clean = read_csv(root / "cleaned/metadata.csv")
            normalized = clean
            original_splits = {}
        for row, norm in zip(clean, normalized):
            filename = row["filename"]
            source_name = row["source_filename"]
            if (norm["filename"], norm["source_filename"]) != (filename, source_name):
                raise ValueError(f"Normalization changed pairing: {source_name}")
            folder = "data_16k" if source == "hf" else "audio"
            audio_path = root / "cleaned" / folder / filename
            original = (raw / "data" / source_name if source == "hf"
                        else root / "cleaned/audio_original" / filename)
            if source == "hf":
                if row["sentence_latin"] != originals[source_name]["sentence_latin"]:
                    raise ValueError(f"HF transcript does not match source: {source_name}")
                for folder in ("data", "data_16k"):
                    if digest(root / "cleaned" / folder / filename) != digest(raw / folder / source_name):
                        raise ValueError(f"HF cleaned copy differs from source: {source_name}")
            audio_hash, duration = audio_info(audio_path)
            text = norm["sentence_latin"].strip()
            if not text or duration < 2 or len(row["sentence_latin"].split()) < 4:
                raise ValueError(f"Invalid cleaned recording: {source_name}")
            if abs(duration - float(row["duration"])) > 0.002:
                raise ValueError(f"Manifest duration differs from audio: {source_name}")
            with wave.open(str(original), "rb") as wav:
                if abs(wav.getnframes() / wav.getframerate() - duration) > 0.002:
                    raise ValueError(f"Original duration differs: {source_name}")
            sample_id = f"{source}_{Path(filename).stem}"
            records.append({
                "sample_id": sample_id, "filename": sample_id + ".wav",
                "source": source, "source_filename": source_name,
                "source_split": original_splits.get(source_name, "unsplit"),
                "text": text, "sentence_latin": text,
                "source_sentence_latin": row.get("source_sentence_latin", row["sentence_latin"]),
                "sentence_cyrillic": row.get("sentence_cyrillic", ""),
                "duration": duration, "word_count": len(text.split()),
                "snr_original": row.get("snr_original", ""), "snr_16k": row.get("snr_16k", ""),
                "audio_sha256": audio_hash, "_audio": audio_path, "_original": original,
            })
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate sample IDs")
    return records


def assign_splits(records, test_fraction, seed):
    # Union both exact PCM duplicates and normalized transcript duplicates,
    # including chains across corpora. A group always has one split everywhere.
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    audio_texts = {}
    for i, row in enumerate(records):
        key = normalized_key(row["text"])
        if row["audio_sha256"] in audio_texts and audio_texts[row["audio_sha256"]] != key:
            raise ValueError(f"Identical audio has conflicting transcripts: {row['sample_id']}")
        audio_texts[row["audio_sha256"]] = key
        for kind, value in (("audio", row["audio_sha256"]), ("text", key)):
            if (kind, value) in seen:
                parent[find(i)] = find(seen[kind, value])
            seen[kind, value] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(records[i])
    targets = {s: round(sum(r["source"] == s for r in records) * test_fraction) for s in CORPORA}
    selected = Counter()
    pending = []
    for group in groups.values():
        if any(r["source_split"] == "test" for r in group):
            for row in group:
                row["split"] = "test"
                selected[row["source"]] += 1
        else:
            pending.append(group)
    random.Random(seed).shuffle(pending)
    for group in pending:
        counts = Counter(r["source"] for r in group)
        before = sum(abs(selected[s] - targets[s]) for s in CORPORA)
        after = sum(abs(selected[s] + counts[s] - targets[s]) for s in CORPORA)
        split = "test" if after < before else "train"
        for row in group:
            row["split"] = split
        if split == "test":
            selected.update(counts)
    for source in CORPORA:
        if {r["split"] for r in records if r["source"] == source} != {"train", "test"}:
            raise ValueError(f"Cannot create both splits for {source}")


def build(stage, records, config):
    combined = []
    pcm_seen = set()
    for source, name in CORPORA.items():
        rows = []
        for record in (r for r in records if r["source"] == source):
            row = {k: v for k, v in record.items() if not k.startswith("_")}
            row["audio_path"] = f"{row['split']}/audio/{row['filename']}"
            destination = stage / name / row["audio_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["_audio"], destination)
            if source == "hf":
                row["original_audio_path"] = f"sources/data/{row['source_filename']}"
            else:
                row["original_audio_path"] = f"{row['split']}/audio_original/{row['filename']}"
                original = stage / name / row["original_audio_path"]
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record["_original"], original)
            rows.append(row)
            if row["audio_sha256"] not in pcm_seen:
                pcm_seen.add(row["audio_sha256"])
                merged = dict(row)
                merged["original_audio_path"] = f"../{name}/{row['original_audio_path']}"
                combined.append(merged)
                copied = stage / "karakalpak-combined" / row["audio_path"]
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record["_audio"], copied)
        write_outputs(stage / name, rows, config)
    write_outputs(stage / "karakalpak-combined", combined, config)
    return combined


def write_outputs(root, rows, config):
    rows.sort(key=lambda row: row["sample_id"])
    for split in ("train", "test"):
        write_csv(root / f"{split}.csv", [row for row in rows if row["split"] == split], TRAIN_FIELDS)
    write_csv(root / "metadata.csv", rows, METADATA_FIELDS)
    write_csv(root / "provenance/metadata.csv", rows)
    summary = {**config, "counts": dict(Counter(row["split"] for row in rows)),
               "sources": dict(Counter(row["source"] for row in rows)),
               "hours": round(sum(float(row["duration"]) for row in rows) / 3600, 3)}
    (root / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "README.md").write_text(
        "# " + root.name + "\n\n"
        "Read `train.csv` or `test.csv`; resolve `audio_path` relative to this folder.\n"
        "Split CSV columns: `audio_path`, `text`, `duration` (seconds).\n"
        "`train/audio/` and `test/audio/` contain mono 16 kHz PCM16 WAVs.\n"
        "`text` is the prepared Latin transcript. Detailed source records are in `provenance/metadata.csv`.\n"
        "`metadata.csv` is the union of both splits with a `split` column.\n"
        "`split_summary.json` records the seed and counts.\n"
        "Assignments are shared across source and combined datasets. Exact PCM audio and normalized\n"
        "transcript duplicates stay together. Original HF test recordings remain in test.\n"
        "No speaker IDs are available; these are recording/transcript groups, not speaker splits.\n"
        "Combined audio files are independent copies; source folders remain separate.\n"
    )


def audit(base=DATA, allow_staged_originals=False):
    assignments = {}
    audio_splits = {}
    text_splits = {}
    source_rows = []
    combined_rows = []
    summaries = {}
    for name in [*CORPORA.values(), "karakalpak-combined"]:
        root = base / name
        rows = read_csv(root / "provenance/metadata.csv")
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate sample IDs in {root}")
        public_metadata = [{key: row[key] for key in METADATA_FIELDS} for row in rows]
        if read_csv(root / "metadata.csv") != public_metadata:
            raise ValueError(f"Public metadata differs from provenance: {root}")
        for split in ("train", "test"):
            subset = read_csv(root / f"{split}.csv")
            expected_subset = [{key: row[key] for key in TRAIN_FIELDS}
                               for row in rows if row["split"] == split]
            if not subset or subset != expected_subset:
                raise ValueError(f"Invalid {split} manifest in {root}")
            expected = {Path(row["audio_path"]).name for row in subset}
            if expected != {path.name for path in (root / split / "audio").glob("*.wav")}:
                raise ValueError(f"Audio/manifest mismatch: {root / split}")
        for row in rows:
            path = root / row["audio_path"]
            audio_hash, duration = audio_info(path)
            if audio_hash != row["audio_sha256"] or abs(duration - float(row["duration"])) > 0.002:
                raise ValueError(f"Audio content mismatch: {path}")
            original = root / row["original_audio_path"]
            if not allow_staged_originals:
                with wave.open(str(original), "rb") as wav:
                    if abs(wav.getnframes() / wav.getframerate() - duration) > 0.002:
                        raise ValueError(f"Original audio duration mismatch: {original}")
            if row["source_split"] == "test" and row["split"] != "test":
                raise ValueError("Published HF test recording entered training")
            for mapping, key in ((assignments, row["sample_id"]),
                                 (audio_splits, audio_hash), (text_splits, normalized_key(row["text"]))):
                if key in mapping and mapping[key] != row["split"]:
                    raise ValueError(f"Train/test leakage: {row['sample_id']}")
                mapping[key] = row["split"]
        summaries[name] = dict(Counter(row["split"] for row in rows))
        if name == "karakalpak-combined":
            combined_rows = rows
        else:
            source_rows.extend(rows)
    expected = {}
    for row in source_rows:
        expected.setdefault(row["audio_sha256"], row)
    actual = {row["audio_sha256"]: row for row in combined_rows}
    if len(actual) != len(combined_rows) or set(actual) != set(expected):
        raise ValueError("Combined dataset is not the deduplicated source union")
    for key, row in actual.items():
        if any(row[field] != expected[key][field] for field in FIELDS if field != "original_audio_path"):
            raise ValueError(f"Combined metadata differs from source: {row['sample_id']}")
    return summaries


def simplify_csv():
    exports = []
    for name in [*CORPORA.values(), "karakalpak-combined"]:
        root = DATA / name
        detail = root / "provenance/metadata.csv"
        rows = read_csv(detail if detail.exists() else root / "metadata.csv")
        if not rows or set(rows[0]) != set(FIELDS):
            raise ValueError(f"Expected complete provenance before simplification: {root}")
        for split in ("train", "test"):
            current = read_csv(root / f"{split}.csv")
            expected = [row for row in rows if row["split"] == split]
            if current != expected and current != [{k: row[k] for k in TRAIN_FIELDS} for row in expected]:
                raise ValueError(f"Split differs from provenance; refusing to overwrite: {root / split}")
        config = json.loads((root / "split_summary.json").read_text())
        exports.append((root, rows, config))
    for root, rows, config in exports:
        # Retain complete pairing records before reducing the public manifests.
        write_outputs(root, rows, config)


def verify_cleanup():
    # Deletion requires recovery archives and byte-identical replacements.
    import subprocess
    import zlib
    root = DATA / CORPORA["mendeley"]
    archives = root / "sources"
    expected = {"DATASET_v1.7z": "97ba2d6f457e3710cd305208aea1196b5c798a544e1b6c62409ae19f2b230918",
                "DATASET_version2.7z": "03ddca8106f8e6247e556665381800eb458f58e2c4c6947bae0b99b290c6d91b"}
    crcs = []
    for name, sha in expected.items():
        path = archives / name
        if digest(path) != sha:
            raise ValueError(f"Archive checksum mismatch: {path}")
        subprocess.run(["7z", "t", str(path)], check=True, stdout=subprocess.DEVNULL)
        listing = subprocess.check_output(["7z", "l", "-slt", str(path)], text=True)
        entries = {}
        filename = ""
        for line in listing.splitlines():
            if line.startswith("Path = "):
                filename = Path(line[7:]).name
            elif line.startswith("CRC = ") and filename.endswith(".wav"):
                entries[filename] = line[6:]
        crcs.append(entries)
    if not crcs[0] or crcs[0] != crcs[1]:
        raise ValueError("Mendeley archives differ; cannot remove version 2")
    manifest = read_csv(root / "aligned/manifest.csv")
    if {row["filename"] for row in manifest} != set(crcs[0]):
        raise ValueError("Mendeley manifest/archive filenames differ")
    for row in manifest:
        path = archives / "version2_extracted/DATASET" / row["filename"]
        if f"{zlib.crc32(path.read_bytes()):08X}" != crcs[0][row["filename"]]:
            raise ValueError(f"Extracted audio does not match recovery archive: {path}")
    for row in read_csv(root / "cleaned/metadata.csv"):
        for aligned, cleaned in (("audio_original", "audio_original"), ("audio_16k", "audio")):
            if digest(root / "aligned" / aligned / row["source_filename"]) != digest(root / "cleaned" / cleaned / row["filename"]):
                raise ValueError(f"Mendeley intermediate differs: {row['source_filename']}")


def reorganize_sources():
    hf = DATA / CORPORA["hf"]
    mendeley = DATA / CORPORA["mendeley"]
    sources = hf / "sources"
    sources.mkdir(exist_ok=True)
    for name in ("data", "data_16k", "data.csv", "train.csv", "test.csv"):
        (hf / name).rename(sources / name)
    (hf / "README.md").rename(sources / "publisher_layout_legacy.md")
    for split in ("train", "test"):
        if (hf / split).is_dir():
            (hf / split).rename(sources / ("legacy_" + split))
    for root in (hf, mendeley):
        provenance = root / "provenance"
        provenance.mkdir(exist_ok=True)
        if root == mendeley:
            (root / "README.md").rename(provenance / "publisher_layout_legacy.md")
        legacy = root / "cleaned"
        for path in list(legacy.glob("*.csv")):
            path.rename(provenance / ("prepared_" + path.name))
    for path in list((mendeley / "aligned").iterdir()):
        if path.is_file():
            path.rename(mendeley / "provenance" / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--simplify-csv", action="store_true",
                        help="Reduce existing public CSVs while retaining detailed provenance")
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("test fraction must be between zero and one")
    marker = DATA / "karakalpak-combined/split_summary.json"
    if args.simplify_csv:
        if not marker.exists():
            parser.error("Organize the datasets before simplifying existing CSVs")
        simplify_csv()
        print(json.dumps(audit(), indent=2))
        return
    if args.audit or marker.exists():
        if marker.exists() and not args.audit:
            previous = json.loads(marker.read_text())
            if previous["seed"] != args.seed or previous["test_fraction"] != args.test_fraction:
                parser.error("Existing split settings differ; preserve the current dataset and build a new export explicitly")
        print(json.dumps(audit(), indent=2))
        return
    stage = DATA / ".organize-staging"
    if stage.exists():
        parser.error(f"Staging folder exists; inspect it before retrying: {stage}")
    config = {"seed": args.seed, "test_fraction": args.test_fraction,
              "grouping": ["PCM SHA256", "normalized transcript"],
              "preserve_published_hf_test": True,
              "speaker_disjoint": False, "combined_deduplication": "PCM SHA256"}
    print("Checking paired inputs and hashing audio...", flush=True)
    records = load_inputs()
    assign_splits(records, args.test_fraction, args.seed)
    if args.cleanup:
        print("Verifying archives and duplicate folders before cleanup...", flush=True)
        verify_cleanup()
    snapshot = []
    for name in CORPORA.values():
        for path in sorted((DATA / name).rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink: {path}")
            if path.is_file():
                snapshot.append({"path": str(path.relative_to(DATA)), "bytes": path.stat().st_size,
                                 "sha256": digest(path)})
    print("Building independent source and combined audio copies...", flush=True)
    build(stage, records, config)
    audit(stage, allow_staged_originals=True)
    # Verify every replacement byte-for-byte before any legacy deletion.
    for record in records:
        root = stage / CORPORA[record["source"]]
        path = root / record["split"] / "audio" / record["filename"]
        if digest(path) != digest(record["_audio"]):
            raise ValueError(f"Staged audio differs: {path}")
        if record["source"] == "mendeley":
            original = root / record["split"] / "audio_original" / record["filename"]
            if digest(original) != digest(record["_original"]):
                raise ValueError(f"Staged original differs: {original}")
    # Reject unexpected export files before relocating any legacy source paths.
    for name in CORPORA.values():
        root = DATA / name
        for generated in ("metadata.csv", "split_summary.json"):
            if (root / generated).exists():
                raise ValueError(f"Unexpected existing export: {root / generated}")
    combined_root = DATA / "karakalpak-combined"
    if combined_root.exists():
        raise ValueError(f"Unexpected existing combined folder: {combined_root}")
    reports = DATA / "organization"
    reports.mkdir(exist_ok=True)
    (reports / "before.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    # Move metadata first; install replacements before deleting legacy audio.
    reorganize_sources()
    for name in [*CORPORA.values(), "karakalpak-combined"]:
        destination = DATA / name
        destination.mkdir(exist_ok=True)
        for path in (stage / name).iterdir():
            target = destination / path.name
            if path.name == "provenance" and target.is_dir():
                for child in path.iterdir():
                    if (target / child.name).exists():
                        raise ValueError(f"Refusing to overwrite provenance: {target / child.name}")
                    child.rename(target / child.name)
                path.rmdir()
                continue
            if target.exists():
                raise ValueError(f"Refusing to overwrite unexpected path: {target}")
            path.rename(target)
    summaries = audit()
    deleted = []
    if args.cleanup:
        # Metadata has been relocated; only these verified legacy audio trees remain.
        targets = [DATA / CORPORA["hf"] / "cleaned", DATA / CORPORA["mendeley"] / "cleaned",
                   DATA / CORPORA["mendeley"] / "aligned",
                   DATA / CORPORA["mendeley"] / "sources/version2_extracted",
                   DATA / CORPORA["mendeley"] / "sources/DATASET_version2.7z"]
        targets += [p for p in DATA.rglob(".ipynb_checkpoints") if not any(t in p.parents for t in targets)]
        targets += [p for p in DATA.rglob(".DS_Store") if not any(t in p.parents for t in targets)]
        for path in targets:
            if not path.exists():
                continue
            files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
            deleted.append({"path": str(path.relative_to(DATA)), "files": len(files),
                            "bytes": sum(p.stat().st_size for p in files)})
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    shutil.rmtree(stage)
    result = {**config, "counts": summaries, "input_recordings": len(records),
              "deleted": deleted, "removed_bytes": sum(r["bytes"] for r in deleted)}
    (reports / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
