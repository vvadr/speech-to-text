"""Audit Mendeley pairs against the authoritative v1 archive, or recover raw audio.

The organized dataset lives in train/ and test/. Version 2's CSV does not
match the audio filenames; never substitute kaa0/kaa1 or align by row order.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path

from organize_datasets import audit, digest, read_csv

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/karakalpak-mendeley"
ARCHIVE = CORPUS / "sources/DATASET_v1.7z"
PROVENANCE = CORPUS / "provenance"
EXPECTED_SHA256 = "97ba2d6f457e3710cd305208aea1196b5c798a544e1b6c62409ae19f2b230918"


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).lower().split())


def archive_mapping():
    if digest(ARCHIVE) != EXPECTED_SHA256:
        raise ValueError("Mendeley v1 archive SHA256 does not match the verified source")
    content = subprocess.check_output(["7z", "e", "-so", str(ARCHIVE), "train.csv"])
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if not rows or set(rows[0]) != {"filename", "text"}:
        raise ValueError("Unexpected publisher CSV schema")
    pairs = {}
    counts = Counter()
    for row in rows:
        name, text = row["filename"], row["text"].strip()
        if not text or Path(name).name != name:
            raise ValueError(f"Invalid publisher row: {name}")
        if name in pairs and pairs[name] != text:
            raise ValueError(f"Conflicting publisher transcripts: {name}")
        pairs[name] = text
        counts[name] += 1
    return pairs, counts


def reviewed_expansions():
    decisions = defaultdict(list)
    for row in read_csv(PROVENANCE / "abbreviation_review.csv"):
        if row["verified"].strip().lower() not in {"yes", "true", "1"}:
            continue
        if not row["evidence"].strip() or not row["spoken_form"].strip():
            raise ValueError(f"Incomplete review: {row['source_filename']}")
        decisions[row["source_filename"]].append(row)
    return decisions


def expected_text(text, decisions):
    result = normalize(text)
    for decision in decisions:
        abbreviation = normalize(decision["abbreviation"])
        spoken = normalize(decision["spoken_form"])
        result, count = re.subn(rf"(?<!\w){re.escape(abbreviation)}(?!\w)", spoken, result)
        if count != 1:
            raise ValueError(f"Expected one reviewed abbreviation: {decision['source_filename']}")
    return result


def audit_mendeley():
    pairs, counts = archive_mapping()
    manifest = read_csv(PROVENANCE / "manifest.csv")
    if len(manifest) != len(pairs) or {row["filename"] for row in manifest} != set(pairs):
        raise ValueError("Provenance filenames differ from the authoritative archive")
    for row in manifest:
        if row["sentence_latin"] != pairs[row["filename"]] or int(row["source_csv_row_count"]) != counts[row["filename"]]:
            raise ValueError(f"Provenance mapping differs: {row['filename']}")
    decisions = reviewed_expansions()
    rows = read_csv(PROVENANCE / "metadata.csv")
    if len(rows) != len(pairs) or {r["source_filename"] for r in rows} != set(pairs):
        raise ValueError("Organized corpus differs from the complete source mapping")
    for row in rows:
        name = row["source_filename"]
        if row["source_sentence_latin"] != pairs[name]:
            raise ValueError(f"Source transcript differs: {name}")
        if row["text"] != expected_text(pairs[name], decisions[name]):
            raise ValueError(f"Prepared transcript differs from review decisions: {name}")
        with wave.open(str(CORPUS / row["original_audio_path"]), "rb") as audio:
            if audio.getnchannels() not in (1, 2) or audio.getsampwidth() != 2:
                raise ValueError(f"Invalid original/decoded audio: {name}")
    summaries = audit()
    print(json.dumps({"authoritative_pairs": len(pairs), "publisher_rows": sum(counts.values()),
                      "verified_expansions": sum(len(v) for v in decisions.values()),
                      "splits": summaries}, indent=2))


def recover():
    pairs, _ = archive_mapping()
    destination = CORPUS / "sources/recovered"
    if destination.exists():
        raise ValueError(f"Recovery destination exists; inspect it first: {destination}")
    subprocess.run(["7z", "t", str(ARCHIVE)], check=True)
    subprocess.run(["7z", "e", str(ARCHIVE), "*.wav", f"-o{destination}", "-y"], check=True)
    if {path.name for path in destination.glob("*.wav")} != set(pairs):
        raise ValueError("Recovered audio filenames differ from source mapping")
    with (destination / "source_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "text"])
        writer.writeheader()
        writer.writerows({"filename": name, "text": pairs[name]} for name in sorted(pairs))
    print(f"Recovered {len(pairs)} original files into {destination}. "
          "Eleven .wav names contain MP4/AAC; decode before loading them as WAV.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "recover"), default="audit")
    args = parser.parse_args()
    {"audit": audit_mendeley, "recover": recover}[args.stage]()


if __name__ == "__main__":
    main()
