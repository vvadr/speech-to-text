# Dataset and code review

## Fixed

- Public train/test CSVs now contain only `audio_path`, `text`, and `duration`.
  Public `metadata.csv` adds only `split`; full audit records are retained under
  `provenance/metadata.csv`. All loaders and audits use the appropriate schema.

- The tracked `mendeley_pipeline-checkpoint.py` referenced the removed
  `Karakalpak Speech Corpus` layout and old CSV names. It is replaced by
  `scripts/mendeley_pipeline.py`, which audits the organized corpus against the
  retained v1 archive and supports recovery of raw recordings.
- Mendeley `cleaned/train.csv` contained all 2,022 recordings without a test
  split. Both sources and their combined export now have explicit train/test
  manifests and audio folders, sharing the same assignments.
- Source-prefixed names prevent collisions between the sources' original
  `audio_000000.wav` sequences.
- The organizer groups PCM and transcript duplicates globally, retains source
  metadata, validates its staging export, and compares replacement files before
  removing redundant audio. Combined recordings are independent file copies.
- Monday/Tuesday notebook paths now target the retained HF `sources/` snapshot.
  Monday resampling preserves existing SNR columns when refreshing its manifest.
- Wednesday previously recreated duplicate cleaned audio and could leave a
  stale normalized CSV after rebuilding. It now reviews selection against the
  canonical exports without replacing them.
- Thursday wrote a redundant normalized CSV with an accidental pandas index
  column. It now explores combined training text in memory. Its normalization
  preserves Unicode letters rather than deleting unfamiliar alphabet characters.
- Friday's unfinished `merge` definition was a syntax error. The byte-pair merge
  now handles adjacent and overlapping pairs.
- Changed notebook outputs were cleared to prevent displaying results from old
  paths. Dataset notebook autosaves and Finder metadata were removed.

## Validation performed

- Both Mendeley archives passed integrity checks and matched the recorded
  SHA256 values. All 2,022 audio CRCs agreed across archives and extraction.
- All removed HF cleaned audio copies matched their retained source WAVs byte
  for byte; all Mendeley intermediate audio matched the retained prepared copies.
- Before checksums are recorded in `data/organization/before.json`. After
  reorganization, all 6,067 retained HF source files (audio plus manifests)
  matched that snapshot exactly.
- Every prepared audio file was checked for complete PCM data, mono 16 kHz
  format, manifest duration, and PCM checksum. Original audio paths resolve.
- Train/test files exactly match their manifests; their union equals the public metadata.
  Simplified CSVs match the corresponding columns in detailed provenance.
  Neither audio hashes nor normalized transcripts cross the train/test boundary.
- Combined data equals the source union, with consistent text and assignments.
- The Mendeley audit recovered 2,022 unique filename/text pairs from 17,238
  publisher rows and verified all seven recorded abbreviation expansions.
- All Python scripts and notebook code cells parse. The changed notebooks are
  structurally valid. Actual data-loading cells ran for the split loader,
  Monday dataset loader, Wednesday selection, and Thursday training-text loader.
- Friday's merge was checked on adjacent and overlapping input pairs. Expensive
  SNR analysis and audio resampling were not rerun because existing audio and
  measurements were preserved.

## Remaining code limitations

`scripts/ConnectionistTemporalClassification/ctc.py` is an unfinished learning
exercise: `ctc_loss()` prints an intermediate target tensor and returns no loss.
It also creates padding on the log-probability device without ensuring targets
use the same device or dtype. It should not be used for training yet.

No speaker identities are available for either corpus. Audio/transcript groups
prevent those specific forms of leakage; unseen-speaker evaluation requires
additional speaker metadata. SNR values are energy-based estimates rather than
ground-truth quality labels.

Dataset changes are local under ignored `data/`. No commit or push was performed.
