# Notebook workflow and published changes

The preprocessing, analysis, and learning notebooks are grouped under
`notebooks/preprocessingAndFeatureExtraction/`. This directory contains all
eight notebooks previously located directly under `notebooks/`.

The dataset organizer, simplified CSV exports, and Mendeley audit scripts are
described in [dataset_review.md](dataset_review.md) and the
[repository README](../README.md). This guide covers every file changed for the
notebook relocation and documentation update.

## File inventory

All notebook paths below are relative to
`notebooks/preprocessingAndFeatureExtraction/`. The old top-level notebook
paths are removed as part of these moves.

| Notebook | Purpose | Inputs and effects |
| --- | --- | --- |
| [dataset_splits.ipynb](../notebooks/preprocessingAndFeatureExtraction/dataset_splits.ipynb) | Load either source or the combined dataset as a Hugging Face `DatasetDict`; preview transcripts and listen to a training recording. | Reads public train/test CSVs and their `audio_path` WAVs. Uses `Audio(decode=False)` with explicit file paths. Writes no dataset files. |
| [monday_1.ipynb](../notebooks/preprocessingAndFeatureExtraction/monday_1.ipynb) | Prepare and validate HF audio at 16 kHz. | Reads preserved source WAVs and publisher train/test CSVs. Skips valid conversions; otherwise writes temporary WAVs before replacement. Refreshes the source `data.csv`, retaining existing SNR columns. |
| [monday_2.ipynb](../notebooks/preprocessingAndFeatureExtraction/monday_2.ipynb) | Load paired original and 16 kHz source audio. | Reads the source `data.csv`, checks sequential indices and filename order, and constructs an in-memory Hugging Face dataset. Writes no dataset files. |
| [tuesday.ipynb](../notebooks/preprocessingAndFeatureExtraction/tuesday.ipynb) | Compare original/resampled audio using STFT, Mel, filterbank, energy VAD, and SNR. | Reads both source audio folders. Computes SNR for all recordings and updates source `data.csv`. Playback and before/after plots use the same selected recording, initially index 9. |
| [wednesday.ipynb](../notebooks/preprocessingAndFeatureExtraction/wednesday.ipynb) | Review duration/word-count filtering and compare selected recordings with the organized HF export. | Reads source metadata, detailed provenance, and public train/test CSVs. Requires >= 2 seconds and >= 4 words; checks sample counts and split separation. Writes no dataset files. |
| [thursday.ipynb](../notebooks/preprocessingAndFeatureExtraction/thursday.ipynb) | Inspect combined training vocabulary, letters, digits, abbreviations, and normalization. | Reads combined `train.csv`; operates on copies in memory. Preserves Unicode letters and digits. Writes no canonical transcripts or CSVs. |
| [friday_tokenizer.ipynb](../notebooks/preprocessingAndFeatureExtraction/friday_tokenizer.ipynb) | Explore UTF-8 byte tokens, adjacent pair counts, and pair merging. | Uses an embedded sample paragraph. Implements one pair merge; does not train or save a complete tokenizer. This file is relocated without content changes in this update. |
| [cuda_mps_check.ipynb](../notebooks/preprocessingAndFeatureExtraction/cuda_mps_check.ipynb) | Inspect CUDA and Apple MPS availability through PyTorch. | Uses the local PyTorch runtime. The CUDA check falls back to CPU; a separate MPS check chooses MPS or CPU. Writes no files. This file is relocated without content changes in this update. |

Other changed files:

- `README.md`: points to the new notebook location, links this guide, and records
  the combined audio duration and train/test durations.
- `docs/dataset_review.md`: explains publication scope and links the notebook
  inventory. The former statement that no commit or push had occurred is removed.
- `docs/notebook_workflow.md`: this complete inventory, setup, paths, execution
  order, validation, and limitations.

The dataset scripts and CTC exercise are covered below for context; their code
is unchanged by the notebook relocation.

## Setup and data requirements

From the repository root:

```bash
uv sync
uv run jupyter lab
```

Open the notebooks under `notebooks/preprocessingAndFeatureExtraction/` and
select the project Python kernel. Dependencies come from `pyproject.toml` and
`uv.lock`, including pandas, datasets, PyTorch, librosa, matplotlib,
imageio-ffmpeg, and Jupyter.

Audio datasets are local and ignored by Git. A fresh clone does not include
recordings, archives, or CSV manifests, and these notebooks do not download
them. Keep the prepared local `data/` tree described in the repository README.

The six notebooks that load recordings now search the current working
directory and its ancestors for the required `data/` manifest. This supports
starting a kernel from the repository root, `notebooks/`, or the new nested
notebook directory. Missing local data raises a `FileNotFoundError` with the
required manifest name. The tokenizer and device checks have no dataset path.

## Which data to use

Use the organized dataset root when training a model:

| Dataset | Train | Test | Total |
| --- | ---: | ---: | ---: |
| `data/karakalpak-audio-dataset-hf/` | 2,090 | 523 | 2,613 |
| `data/karakalpak-mendeley/` | 1,618 | 404 | 2,022 |
| `data/karakalpak-combined/` | 3,708 | 927 | 4,635 |

Public `train.csv` and `test.csv` contain `audio_path`, `text`, and `duration`
in seconds. Resolve `audio_path` relative to the selected dataset root.
`metadata.csv` combines both splits with an additional `split` column. Full
source identifiers, original transcripts, quality measurements, and checksums
are kept in `provenance/metadata.csv`.

The combined corpus lasts 30,783.409 seconds, approximately 8 h 33 m 3 s:
24,615.091 seconds in train and 6,168.318 seconds in test. All prepared audio is
mono 16 kHz PCM16 WAV. Source-prefixed filenames prevent collisions between
the two source datasets.

The Monday and Tuesday notebooks deliberately analyze the preserved HF source
snapshot under `data/karakalpak-audio-dataset-hf/sources/`. Its `train.csv` and
`test.csv` retain the publisher's original 2,982/50 split and schema; its
`data.csv` and original/resampled audio folders cover all 3,032 recordings.
These source CSVs differ from the simplified public CSVs at the dataset root.

## Suggested order

1. Open `dataset_splits.ipynb` to preview and use the prepared training export.
   Set `dataset_name` to the combined corpus or either separate source.
2. For source audio analysis, run Monday 1, Monday 2, and Tuesday in order.
   Monday 1 can regenerate invalid/missing 16 kHz copies. Tuesday recalculates
   quality measurements and produces before/after visualizations.
3. Use Wednesday to inspect the cleaning rule and verify the prepared HF
   selection. It does not recreate duplicate cleaned audio folders.
4. Use Thursday to explore text from the combined training split. Test
   transcripts remain held out from this vocabulary analysis.
5. Run the tokenizer and device notebooks independently for those exercises.

Run each notebook's cells from the top. Analysis notebooks do not assign new
training splits; assignments are maintained by `scripts/organize_datasets.py`.

## Dataset maintenance

From the repository root:

```bash
.venv/bin/python scripts/organize_datasets.py --audit
.venv/bin/python scripts/mendeley_pipeline.py --stage audit
```

The organizer validates public CSVs against provenance, audio formats and
hashes, split consistency, and the combined source union. Its initial import
accepts the earlier `cleaned/` layout, uses seed 42 and an 80/20 split, and
supports `--cleanup` after verified replacements are installed. Existing
exports are audited rather than assigned new splits. `--simplify-csv` reduces
older public manifests while retaining their detailed provenance.

Mendeley auditing uses the verified v1 archive and its authoritative filename
mapping, including seven recorded abbreviation expansions. `--stage recover`
extracts source files into `sources/recovered/`; it does not replace training
exports. Eleven recovered `.wav` names contain MP4/AAC and require decoding.

The cleanup retained the HF sources and Mendeley v1 recovery archive, removed
verified duplicate audio folders and the redundant v2 archive/extraction, and
saved approximately 2.6 GB net after creating the combined audio copies.
Before checksums and deletion details remain local under `data/organization/`.

## Validation and limitations

Validation on 2026-10-09: all eight moved notebooks passed notebook structure
and Python syntax checks. Data-loading cells in all six dataset notebooks
loaded successfully from both the repository root and the new nested directory
against the actual local manifests. These checks do not regenerate audio, recalculate
all SNR values, or overwrite source CSVs. Existing notebook outputs are retained
as empty cells where they were previously cleared.

The dataset audit passed for all three exports, checking the simplified public
CSVs against detailed provenance and real WAV files. Earlier archive integrity and source-preservation
checks are documented in `dataset_review.md`; source data remains outside the
Git publication.

Audio and normalized transcript groups share split assignments across all
three datasets. The 46 retained original HF test recordings remain in test.
Speaker identities are unavailable, so unseen-speaker evaluation is not
guaranteed. SNR is an energy-based estimate rather than a ground-truth label.

`scripts/ConnectionistTemporalClassification/ctc.py` remains an incomplete
exercise that prints an intermediate target tensor and returns no loss. The
Friday notebook demonstrates pair merging rather than a complete trained
tokenizer. Neither is a finished ASR training implementation.

Only code, notebooks, and documentation are published. `/data/`, notebook
autosaves, virtual environments, credentials, and caches are excluded by
`.gitignore`.
