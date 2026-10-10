# Whisper training: beginner guide

Start with `notebooks/whisper/setup.ipynb`. The workflow is written with ordinary
variables, tables, small functions, and loops. Every implementation file is a
notebook. Each stage imports its libraries and loads missing prerequisites when
opened in a fresh kernel. The runner executes the stages in one Python session.

This project fine-tunes the complete pretrained **Whisper-small** model on
Karakalpak speech: all **241,734,912 parameters** can change. It uses every
training recording after validation is reserved. Source CSVs and WAVs are only
read; model outputs go under the ignored `checkpoints/` folder.

## Run it

From the repository root:

```bash
uv run ipython -c "get_ipython().run_line_magic('run', 'notebooks/whisper/setup.ipynb'); run_notebooks(train=True)"
```

This loads setup, checks the dataset and model, trains for three epochs, then
scores the best checkpoint on validation. For a check without training:

```bash
uv run ipython -c "get_ipython().run_line_magic('run', 'notebooks/whisper/setup.ipynb'); run_notebooks(train=False)"
```

In Jupyter, run the setup cells and enter `run_notebooks(train=True)` when ready to train. Read
the numbered notebooks to follow each step. Each stage imports its own libraries and loads earlier data/model steps when
needed. Training is disabled by default, including when an individual notebook
is opened. The runner requires `train=True` to start learning.
The runner does not rewrite notebook files or saved cell outputs. A failing
code cell stops the workflow immediately.

After a completed experiment, change `RUN_NAME` in setup before training again.
A nonempty experiment folder is refused so an earlier model is not replaced.
Local audio is not included in a fresh clone. `LOCAL_FILES_ONLY = True` uses the
model already cached on this Mac; set it to `False` if weights need downloading.

## Select the correct kernel and see progress

The project environment has pandas and PyTorch installed. Choose
**Whisper (project .venv)** from the notebook's kernel selector. Its interpreter
is `/Users/macbook/Documents/IT/repos/AI & ML/.venv/bin/python`.
A kernel using another Python environment may not have these packages.

Start Jupyter from the project environment if needed:

```bash
uv run jupyter lab
```

The named kernel has been registered inside the project `.venv`. To register it
again after recreating the environment:

```bash
uv run python -m ipykernel install --sys-prefix --name whisper-project --display-name "Whisper (project .venv)"
```

For training with a visual monitor:

1. In setup, choose a fresh `RUN_NAME`, run its cells, and call
   `run_notebooks(train=True)`. The runner checks data/model inputs, then executes
   training (`04_training.ipynb`) and evaluation (`05_evaluation.ipynb`).
2. In another tab, open `07_progress.ipynb` using the named kernel. Change
   `RUN_TO_WATCH` to the same run name and click **Run All**.
3. Rerun the viewer's cells to refresh. The bar shows training-epoch or evaluation
   progress, the loss curve shows weight updates, and the epoch chart appears
   after the first validation pass.

New runs save `loss_history.csv`, so plots work for both notebook and terminal
training. The viewer uses the old terminal log as a fallback for the stopped
first run. A lower training-loss curve does not by itself prove recognition
accuracy. The viewer reads files only and never loads or starts the model.

## Each notebook

The runner uses this order: **setup → data → model → batching → checks → training
→ evaluation**. Checks run before training even though their filename starts
with `06`.

| File | What to read and what it does |
| --- | --- |
| [setup.ipynb](../notebooks/whisper/setup.ipynb) | Imports tools, finds the project, lists editable settings, chooses the device, and defines the short `run_notebooks()` function. |
| [01_data.ipynb](../notebooks/whisper/01_data.ipynb) | Reads train/test CSVs, checks every recording, groups repeated transcripts, and reserves validation from train. |
| [02_model.ipynb](../notebooks/whisper/02_model.ipynb) | Loads the pretrained processor/model, configures the transcription prefix, and verifies that all parameters are trainable. |
| [03_batching.ipynb](../notebooks/whisper/03_batching.ipynb) | Defines `prepare_batch`, builds data loaders from lists of table rows, and checks one real forward loss. |
| [06_checks.ipynb](../notebooks/whisper/06_checks.ipynb) | Uses short assertions to verify split separation, unchanged test membership, valid settings, expected feature shapes, and trainable parameters. |
| [04_training.ipynb](../notebooks/whisper/04_training.ipynb) | Defines `validation_loss` and `train_one_epoch`, prepares the run folder/optimizer, runs three epochs, and saves the best validation checkpoint. These are separate, explained cells. |
| [05_evaluation.ipynb](../notebooks/whisper/05_evaluation.ipynb) | Reloads the best checkpoint, checks saved split membership, generates transcripts, and saves recognition scores and predictions. |
| [07_progress.ipynb](../notebooks/whisper/07_progress.ipynb) | Read-only stage/status, progress bar, training-loss curve, and completed-epoch charts. It never starts training. |

[docs/whisper_training.md](whisper_training.md) is this guide. `README.md` points
to the notebook workflow, and `.gitignore` keeps datasets and checkpoints out of
Git. Existing unrelated repository files are unchanged. No extra Python modules
or notebook-import package are needed.

## The settings in plain language

| Setting | Default | Meaning |
| --- | --- | --- |
| `MODEL_ID` | `openai/whisper-small` | The pretrained model to fine-tune. |
| `RUN_NAME` | `whisper-small-kaa-full-run02` | Name of the output folder; change it for a new experiment. |
| `BATCH_SIZE` | 1 | Recordings processed together in one forward/backward pass. |
| `EPOCHS` | 3 | Complete passes over the training recordings. |
| `LEARNING_RATE` | 0.00001 | Controls how strongly the optimizer changes weights. |
| `ACCUMULATION_STEPS` | 8 | Combine gradients from eight batches before one weight update. |
| `VALIDATION_FRACTION` | 0.10 | Reserve about 10% of original train for checking learning. |
| `RUN_FINAL_TEST` | False | Use validation for scoring; enable final test only after model/settings are selected. |
| `LOCAL_FILES_ONLY` | True | Require locally cached pretrained artifacts. |

The device is selected as CUDA first, then Apple MPS, then CPU. MPS is PyTorch's
Apple GPU backend. CPU works for checks but is much slower for full training.
Gradient checkpointing recomputes intermediate values during backward passes
instead of retaining all of them, reducing memory at the cost of more work.
Training uses float32, AdamW weight decay 0.01, and gradient clipping at norm 1.
There is no learning-rate scheduler, mixed precision, augmentation, or freezing.

## How data becomes a transcript

Whisper-small is an **encoder-decoder Transformer**. The encoder reads the audio;
the decoder predicts the text. It reuses the existing multilingual tokenizer.
The [published model configuration](https://huggingface.co/openai/whisper-small/blob/main/config.json)
records 12 encoder layers, 12 decoder layers, hidden width 768, and 12 attention
heads per layer. Its vocabulary has 51,865 token IDs.

```mermaid
flowchart LR
    A[Mono 16 kHz audio] --> B[80-bin log-Mel features]
    B --> C[Two convolution layers]
    C --> D[12-layer audio encoder]
    D --> E[12-layer text decoder]
    T[Previous text tokens] --> E
    E --> F[Scores for the next token]
    F --> G[Generated transcript]
```

A log-Mel spectrogram describes how audio energy changes across time and
frequency. The processor pads recordings to a 30-second input window, producing
features shaped `[batch, 80, 3000]`. Two convolution layers reduce the time
positions to 1,500. The encoder learns audio context; decoder cross-attention
reads that context while causal self-attention reads preceding text tokens.
The [official model implementation](https://github.com/openai/whisper/blob/main/whisper/model.py)
shows these operations. The decoder's text context is 448 tokens.

A token ID is a number representing a text fragment or a special marker. The
label sequence contains language/task markers, transcript tokens, and an
end-of-transcript marker. Padding becomes `-100` so the loss ignores it. The
model adds its decoder start token when shifting labels, so `prepare_batch`
removes that token once. This matches the
[Whisper fine-tuning recipe](https://huggingface.co/blog/fine-tune-whisper).

`LANGUAGE = "kk"` means **Kazakh** in
[Whisper's language list](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py).
Whisper has no dedicated Karakalpak token; the existing Kazakh proxy is retained
for this experiment. It is not a Karakalpak language code. Compare alternative
prefixes on validation, keeping test scores for final reporting.

Whisper uses sequence-to-sequence cross-entropy, rather than CTC. During training,
it reads the previous correct reference tokens; during recognition, it reads
its own previously generated tokens. This difference is why a forward loss check
alone does not establish recognition quality.

## What happens during training

1. **Check data.** Read all audio, verify mono 16 kHz, finite samples, positive
   duration, and a maximum of 30 seconds. Verify manifest durations within 20 ms
   and paths inside the dataset. This prepared corpus has no exact audio
   duplicates; a duplicate stops the run for review. Repeated normalized
   transcripts stay in the same split. Normalization changes only grouping
   keys, never training text. Overlong audio/text needs matched segmentation.
2. **Reserve validation.** Keep the original 927 test rows untouched. With seed
   42, original train is split into **3,338 train and 370 validation**. Speaker
   IDs are unavailable, so this is not an unseen-speaker guarantee.
3. **Prepare batches.** `prepare_batch` reads only the current batch and makes
   audio features, attention masks, and text labels. The whole corpus's features
   are not stored in RAM. Training rows shuffle; validation rows stay ordered.
4. **Predict and measure loss.** The model predicts token scores, and
   cross-entropy measures disagreement with correct token IDs. Lower loss means
   the model better fits those targets.
5. **Calculate gradients.** `loss.backward()` calculates how each weight affects
   loss. Multiply the batch's mean loss by its real target-token count so the
   accumulation window contains a summed token loss.
6. **Update weights.** Every eight batches, divide gradients by the actual token
   count, clip their norm, and call `optimizer.step()`. The final partial window
   uses its own count. Clear gradients before collecting the next window.
7. **Validate and save.** At the end of every epoch, measure validation loss
   without gradients. Average loss over nonpadding target tokens. Save the
   model and processor to `best/` when validation improves; test never selects
   the checkpoint.
8. **Evaluate recognition.** After all three epochs, reload `best/` and generate
   transcripts for every validation recording. WER counts word substitutions,
   deletions, and insertions divided by reference words; CER uses characters.
   Fractions are multiplied by 100 for display. Insertions can make WER exceed
   100%. Scoring keeps case, punctuation, and diacritics, with ordinary whitespace
   handling. Predictions and references are saved for review.

Final test scoring is a separate decision after choosing settings on validation:
set `RUN_FINAL_TEST = True` and run the evaluation stage in a session with the
preceding data/settings loaded. It scores all 927 test recordings. Generation is
greedy by default and limited to 444 new tokens; reaching the cap may truncate
an output, so inspect longer predictions.

## Files produced by a full run

```text
checkpoints/
  whisper-small-kaa-full-run01.log   # background terminal output
  whisper-small-kaa-full-run01.pid   # background process ID
  whisper-small-kaa-full-run01/
    settings.json                   # model, device, hyperparameters, row counts
    train_split.csv                 # exact training membership
    validation_split.csv
    test_split.csv
    progress.json                   # training/validation/evaluation stage
    loss_history.csv                # each update, for live notebook plots
    history.json                    # completed epochs and validation losses
    best/                           # selected model and processor
    validation_metrics.json         # after training and validation decoding
    test_metrics.json               # only if final testing is enabled
```

The log/PID files are produced by the background launch, not by notebook code.
`progress.json` updates after each optimizer update. `best/` appears after the
first completed epoch improves validation loss. These are inference checkpoints;
optimizer/RNG state is not saved, and exact training resume is not implemented.
Keep source data unchanged between training and evaluation. Saved split rows are
checked before recognition scores are reported. There is no distributed or
multi-GPU support.

To follow this run from a terminal:

```bash
tail -f checkpoints/whisper-small-kaa-full-run01.log
```

## Verification

On 2026-10-10, all eight notebook structures and code syntax passed validation.
After adding imports/prerequisite loading, all eight notebooks ran successfully
in fresh project-environment IPython sessions with training disabled. The named
project kernel was registered, and the progress viewer displayed the stopped run.
The complete workflow ran with training disabled on the **Apple GPU**, using the
actual dataset and pretrained small checkpoint. Data, split, batch, finite-loss,
and all-parameters-trainable checks passed. A separate regression verified that
variable token counts and the final partial accumulation window produce the
same update as one combined token batch. The source datasets were not modified.

A small, randomly initialized Whisper model and real recordings are used for
an affordable external contract check of training, checkpoint save/reload, and
recognition. This check is not the full-model training run or an ASR benchmark.
The long full-model run is tracked separately through its log and progress file.

## Stopped full-model run

The first full run was stopped at the user's request on 2026-10-10. Its final
recorded progress was epoch 1, batch **864 of 3,338**, with **108 optimizer
updates**. The progress status is now `stopped`; the training process is gone.
It had not completed the first epoch, so no model checkpoint or validation score
had been saved. The next configured run is `whisper-small-kaa-full-run02` and
training is disabled until explicitly requested. This is a new training run,
not an exact continuation of the stopped weights.

The previous output/log remains under `checkpoints/whisper-small-kaa-full-run01*`
for inspection. Open `07_progress.ipynb` to visualize its stopped status and loss.
