# Karakalpak ASR datasets

The two source datasets stay separate. A third folder combines their **cleaned**
recordings for model training. All prepared audio is mono 16 kHz PCM16 WAV.

| Folder under `data/` | Train | Test | Total |
| --- | ---: | ---: | ---: |
| `karakalpak-audio-dataset-hf` | 2,090 | 523 | 2,613 |
| `karakalpak-mendeley` | 1,618 | 404 | 2,022 |
| `karakalpak-combined` | 3,708 | 927 | 4,635 |

```text
data/
  karakalpak-audio-dataset-hf/
    train.csv, test.csv, metadata.csv, split_summary.json
    train/audio/
    test/audio/
    sources/                  # untouched source WAVs, 16 kHz copies, publisher CSVs
    provenance/               # prior selection and normalization records
  karakalpak-mendeley/
    train.csv, test.csv, metadata.csv, split_summary.json
    train/audio/, train/audio_original/
    test/audio/, test/audio_original/
    sources/DATASET_v1.7z      # verified recovery archive with correct text mapping
    provenance/               # source mapping, measurements, abbreviation decisions
  karakalpak-combined/
    train.csv, test.csv, metadata.csv, split_summary.json
    train/audio/
    test/audio/
    provenance/metadata.csv   # detailed source records and checksums
  organization/               # before checksums and cleanup report
```

The training CSVs contain only three columns:

| Column | Meaning |
| --- | --- |
| `audio_path` | WAV path relative to the dataset folder |
| `text` | Prepared Latin transcript |
| `duration` | Audio duration in seconds |

`metadata.csv` combines both splits and adds only `split`. Detailed source
filenames, original transcripts, Cyrillic text, quality measurements, and audio
checksums are kept in `provenance/metadata.csv` for auditing.

Start with [notebooks/preprocessingAndFeatureExtraction/dataset_splits.ipynb](notebooks/preprocessingAndFeatureExtraction/dataset_splits.ipynb) to load
any folder as a Hugging Face `DatasetDict` and listen to a matched training row.
The Monday and Tuesday notebooks analyze the preserved HF source snapshot;
Wednesday reviews its cleaning rule; Thursday explores combined training text.
All eight notebooks live under `notebooks/preprocessingAndFeatureExtraction/`.
See [docs/notebook_workflow.md](docs/notebook_workflow.md) for each notebook,
setup, execution order, dataset paths, and limitations.

The combined recordings total **8 hours, 33 minutes, 3 seconds**:
6 hours, 50 minutes, 15 seconds in train and 1 hour, 42 minutes, 48 seconds in test.

## Splits and cleaning

The split is 80% train / 20% test, rounded per source, with seed 42. Assignments
are identical in the separate and combined datasets. Exact audio duplicates
and normalized transcript duplicates are grouped before splitting. The 46
retained recordings from the original HF test set remain in test. Speaker IDs
were not supplied, so these splits cannot guarantee unseen speakers.

The existing cleaning rule keeps recordings with duration >= 2 seconds **and**
at least four words: 2,613 of 3,032 HF recordings and all 2,022 Mendeley
recordings. All 3,032 original HF pairs are retained under `sources/`.
No exact duplicate audio was found among the 4,635 prepared recordings.

Mendeley text uses the authoritative v1 filename mapping and seven previously
reviewed abbreviation expansions. The v2 CSV uses mismatched filenames and is
excluded from training. Its archive and extracted audio were removed after
archive integrity, checksums, and all audio CRCs matched the retained v1 archive.
Duplicate intermediate audio and notebook autosaves were also removed.
The net dataset size decreased by approximately **2.6 GB** after creating the
independent combined audio copies.

## Maintenance

```bash
.venv/bin/python scripts/organize_datasets.py --audit
.venv/bin/python scripts/mendeley_pipeline.py --stage audit
```

To simplify CSVs from an earlier organized export, run
`.venv/bin/python scripts/organize_datasets.py --simplify-csv`.

The organizer imports the previous local `cleaned/` layout on its first run;
`--cleanup` removes verified redundant files after publication and audit. An
already organized dataset is audited on subsequent runs, preserving assignments.
It refuses a different seed or test ratio when outputs already exist.

To recover original Mendeley source containers when needed:

```bash
.venv/bin/python scripts/mendeley_pipeline.py --stage recover
```

Recovery writes `sources/recovered/` without changing training exports. Eleven
source files have `.wav` names but contain MP4/AAC and require decoding before
WAV loading. The training and preserved decoded audio are valid WAV files.

Local datasets remain ignored by Git. See
[docs/dataset_review.md](docs/dataset_review.md) for the code review and validation.

## Whisper fine-tuning

Start with [notebooks/whisper/setup.ipynb](notebooks/whisper/setup.ipynb).
The code uses plain settings, tables, small functions, and loops. The runner
executes all stages in one Python session, checks the data/model, trains every
Whisper-small parameter for three epochs, and evaluates the selected checkpoint.

```bash
uv run ipython -c "get_ipython().run_line_magic('run', 'notebooks/whisper/setup.ipynb'); run_notebooks(train=True)"
```

Choose the **Whisper (project .venv)** kernel. Use `run_notebooks(train=False)`
for checks without training; training is disabled by default. Each stage imports
its libraries and loads missing prerequisite steps. Read the numbered
notebooks to follow the steps, and see [docs/whisper_training.md](docs/whisper_training.md)
for each file, beginner explanations, architecture, settings, training stages,
and progress logs. Source datasets stay read-only; model outputs remain local
under the ignored `checkpoints/` folder. Change `RUN_NAME` before a new experiment.

Open [07_progress.ipynb](notebooks/whisper/07_progress.ipynb) in another tab to
visualize the current stage, progress bar, and loss curves. Choose the same run
name and rerun its cells to refresh. The previous full run was stopped before
its first checkpoint; the next configured run is `whisper-small-kaa-full-run02`.
