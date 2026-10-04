# Mendeley Karakalpak Speech Corpus pipeline

This pipeline uses the published [Mendeley version 1](https://data.mendeley.com/datasets/2th8jvft8f/1) `train.csv` to pair text with recordings by **filename**. The version 2 CSV has generated `sentence_kaa0...` names that do not match its audio. Version 1 has 17,238 CSV rows but only 2,022 distinct filename/text pairs. Repeated filenames have the same text, and their audio CRCs match version 2 and the locally extracted recordings. The version 2 CSV also changes three words to crude replacements; its row order is not used to assign transcripts.

The source archives and extracted version 2 files stay in `data/Karakalpak Speech Corpus/`. The pipeline writes to `data/Karakalpak Speech Corpus/aligned/`. Everything under `data/` is local and Git ignored.

## Outputs

| Path under `aligned/` | Meaning |
| --- | --- |
| `source_train_v1.csv` | Exact publisher version 1 CSV, including duplicate rows. Reference only. |
| `manifest.csv` | One authoritative source filename/transcript pair per recording. |
| `train.csv` | Two-column `filename,text` CSV with exactly one row per source audio file. |
| `provenance.json` | Archive digests, pair counts, CRC checks, and source container counts. |
| `data/` | 2,022 original WAVs; 11 mislabeled AAC files are decoded to WAV. Original stereo WAVs remain stereo. |
| `data_16k/` | Matching 16 kHz mono PCM WAVs. |
| `data.csv` | Paired source filenames and transcripts with duration and original/16 kHz SNR estimates. |
| `abbreviation_expansions.csv` | Per-recording audio review decisions and evidence. |
| `cleaned/data.csv` | Rows retained by the existing duration and word-count rule, renamed in sequential order. |
| `cleaned/data/`, `cleaned/data_16k/` | Corresponding sequentially named audio in both forms. |
| `cleaned/removed_annotations.csv` | **Training transcript CSV**: lowercase, punctuation removed, spaces collapsed, and only audio-supported abbreviations expanded. Includes `source_filename` and original transcript for traceability. |
| `cleaned/train.csv` | Minimal two-column `filename,text` training manifest matched to `cleaned/data_16k/`. |

The rule removes a pair when `duration < 2` seconds or `word_count < 4`. This corpus retains all 2,022 pairs. It contains about 5.07 hours of audio. There are 23 stereo WAV sources and 11 MP4/AAC files with `.wav` extensions. A missing SNR means no separate quiet/speech region was found; it does not drop the pair.

## Run

From the repository root, with the existing project environment and `7z` available:

```bash
.venv/bin/python scripts/mendeley_pipeline.py
```

The stages can also run separately with `--stage align|audio|features|clean|audit`. `--workers 6` controls audio preparation concurrency. The script checks both archive SHA-256 values, source and extracted audio CRCs, row mapping, WAV properties, normalized text, and exact cleaned audio copies. Re-running it preserves the local `abbreviation_expansions.csv` review table.

Open `notebooks/mendeley_review.ipynb` to play the first matched recording and inspect any other pair, view original versus 16 kHz audio features, and play each abbreviation decision.

## Abbreviation decisions

The publisher text is mostly lowercase and already stripped of punctuation. Review focused on `tb`, `t b`, `xbr`, and `qr`. Local Karakalpak Wav2Vec2 CTC inference on the matching recordings supported seven full forms, which are listed in `abbreviation_expansions.csv` with the source filename and evidence. When the model suggested spoken letters or was unclear, the text stays as supplied. The review was model assisted; the decisions are available for a Karakalpak speaker to check by playing each recording in the review notebook.

The filename mapping is publisher supplied and archive verified. This does not prove that every word in all 2,022 transcripts matches human listening; the review notebook makes spot checks and abbreviation decisions easy to inspect. No speaker identifiers are present in the local archives, so this pipeline does not create a speaker independent train/validation/test split.
