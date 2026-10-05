# Mendeley Karakalpak speech corpus

## Start here

The dataset ready for training is in **`data/karakalpak-mendeley/cleaned/`**:

- `train.csv` has exactly one `filename,text` row for each recording.
- `audio/` has the matching **16 kHz mono WAV** files. For example, the first CSV row names `audio_000000.wav`, which is `audio/audio_000000.wav`.
- `metadata.csv` has the same cleaned text plus the original filename, source text, duration, SNR estimates, and other traceability fields.
- `audio_original/` has the same 2,022 recordings before resampling, under the same cleaned filenames.

The training CSV and its audio sit together. The source and intermediate folders are separate:

```text
data/
├── karakalpak-audio-dataset-hf/    # Separate Hugging Face dataset
└── karakalpak-mendeley/
    ├── README.md                   # Local quick guide
    ├── cleaned/                    # USE THIS FOR TRAINING
    │   ├── train.csv               # filename,text
    │   ├── audio/                  # 16 kHz mono WAVs named by train.csv
    │   ├── metadata.csv            # Cleaned text and source traceability
    │   ├── audio_original/         # Matching original or decoded WAVs
    ├── aligned/                    # Intermediate pairs and measurements
    │   ├── source_pairs.csv        # Original filename,text mapping
    │   ├── manifest.csv
    │   ├── selection.csv           # Selected rows before text normalization
    │   ├── audio_original/
    │   ├── audio_16k/
    │   ├── measurements.csv
    │   ├── abbreviation_review.csv
    │   ├── publisher_train_v1.csv  # Publisher rows, including duplicates
    │   └── provenance.json
    └── sources/                    # Original Mendeley archives and extraction
        ├── DATASET_v1.7z
        ├── DATASET_version2.7z
        └── version2_extracted/
```

Everything under `data/` stays local and Git ignored.

## How the pairs were selected

The [published Mendeley version 1](https://data.mendeley.com/datasets/2th8jvft8f/1) `train.csv` has 17,238 rows but only 2,022 distinct filename/transcript pairs. Repeated filenames have the same text. Version 1 filenames match all 2,022 local recordings, and audio CRCs match the version 2 archive. Version 2's CSV uses generated `sentence_kaa0...` names that do not match its audio, so its row order is not used to assign transcripts.

The pipeline decoded 11 MP4/AAC recordings that had `.wav` names, preserved 23 stereo WAV sources in `audio_original/`, and prepared all 2,022 recordings as 16 kHz mono WAVs. The `(duration < 2) | (word_count < 4)` rule removed no pairs. Total recording duration is about 5.07 hours. A missing SNR means no separate quiet and speech regions were found; it does not remove the pair.

Text was lowercased, punctuation was removed, and whitespace was collapsed. Seven abbreviations were expanded only where local Karakalpak ASR supported the spoken full form. The per-recording decisions and evidence are in `aligned/abbreviation_review.csv`; uncertain cases remain as supplied. This is model-assisted review, and `notebooks/mendeley_review.ipynb` lets a Karakalpak speaker play each recording.

## Rebuild or inspect

From the repository root, with the project environment and `7z` installed:

```bash
.venv/bin/python scripts/mendeley_pipeline.py
```

The stages are `align`, `audio`, `features`, `clean`, and `audit`; `--stage` runs one stage. `--workers 6` controls audio preparation concurrency. The script checks archive SHA-256 values, source and extracted audio CRCs, filename mappings, WAV properties, cleaned text, and copied audio. Re-running preserves the local `aligned/abbreviation_review.csv` decisions.

Open `notebooks/mendeley_review.ipynb` to play the first matched pair, inspect original and 16 kHz audio features, and review abbreviation decisions. The available archives do not include speaker identifiers, so this pipeline does not create a speaker-independent train/validation/test split.
