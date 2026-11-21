# Training Ava

This is the full recipe used to train the model in this repository. The
two scripts in `training/` are the actual scripts from the training run,
not simplified versions.

## What you need

- A GPU. The reference run used a single RTX Pro 6000. The model itself
  is small, but 58,000 utterances over 150 epochs is still a GPU job.
- About 30 GB of free disk for the data preparation step.
- Python with `torch`, `numpy`, `scipy`, `soundfile` and `datasets`
  (for the Neyshekar download).

## Step 1: prepare the data

```bash
python training/prepare_data.py --work stt_v6_work
```

This downloads the Neyshekar dataset (the `shekar-ai/Neyshekar`
parquet mirror on Hugging Face, about 11 GB), decodes the audio, and
extracts log-mel features with the frozen frontend below. It writes:

- `stt_v6_work/features/real_train.pkl` — 58,167 training items
- `stt_v6_work/features/real_dev.pkl` — 1,884 validation items
- `stt_v6_work/features/stats.json` — per-band mean and std, used for
  normalization

The script verifies its own output against the expected item and frame
counts (58,167 / 1,884 items, 32,831,886 / 1,123,843 frames) and refuses
to continue if they do not match. The parquet files and the decoded wavs
are deleted afterwards unless you pass `--keep-parquet` or `--keep-wavs`.

The frontend is fixed and must not change between training and inference:

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Window | 25 ms Hann, periodic |
| Hop | 10 ms |
| FFT size | 512 |
| Mel bands | 40, HTK scale, 20–8000 Hz |
| Pre-emphasis | 0.97 |
| Log floor | 1e-6 |
| Vocabulary | 47 characters: Persian letters, space, ZWNJ, Persian digits ۰–۹ |

## Step 2: train

```bash
python training/train.py --features stt_v6_work/features --epochs 150 --batch 128 --lr 2e-3 --amp
```

The defaults are the recipe from the reference run:

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 2e-3, weight decay 0.01 |
| Schedule | 5-epoch warmup, then cosine decay to 1e-5 |
| Batch | 128, with length bucketing |
| Loss | CTC, blank id 0, computed in fp32 |
| Precision | bf16 autocast (`--amp`) |
| Gradient clipping | 1.0 |
| Augmentation | SpecAugment: 2 frequency masks (up to 8 bands), 2 time masks (up to 40 frames) |
| Early stopping | patience 25 epochs on dev CER |
| Seed | 42 |

Checkpoints go to `ckpt_v6/`: `best.pt` on every improvement,
`last.pt` every epoch, and a periodic checkpoint every 5 epochs.
Resume a run with `--resume ckpt_v6/last.pt`. A quick smoke test
(300 items, 2 epochs) is `--smoke`. If you pass `--hf-repo` and set
`HF_TOKEN`, the best checkpoint is also uploaded to a private
Hugging Face repo on every improvement, so a lost session does not
lose the run.

Two details worth knowing:

- The CTC alignment filter drops 2,632 of the 58,167 training items
  (4.5%) whose transcripts are too long for the model's 8x time
  downsampling. That is expected, not an error.
- 23 of the 1,884 dev items are likewise longer than the model can
  align; they are kept in the evaluation set so the numbers stay
  comparable.

## What to expect

In the reference run, dev CER improved steadily and the best
checkpoint came at epoch 116 with a dev CER of 0.2408; training
stopped early at epoch 141. Your run should land in the same
neighborhood if the data preparation verified cleanly. If the first
epochs show a loss that is not finite, the script stops by itself and
writes an emergency checkpoint — that usually means a bad feature
file, so re-run the preparation step with `--verify-only` first.

## Step 3: export (how the published files were made)

The ONNX file in `models/` was exported from `best.pt` with the
frontend folded into the graph: the model takes a raw waveform and
returns logits, so nothing but the wav file is needed at inference
time. It was then quantized to int8 with onnxruntime; the cost of
quantization on the full dev set was +0.0001 CER.

The TFLite file was rebuilt in Keras from the same checkpoint
(BatchNorm folded into the convolutions, PyTorch NCHW weights
transposed to NHWC, the mean/std normalization kept inside the
graph), then converted with full-integer quantization using a
representative set of 200 real dev utterances. Its input is fixed at
(1, 40, 600) log-mel frames, which is the 6-second cap.

The export scripts themselves are not part of this repository; the
two files above are the verified outputs, and `demo/` shows exactly
how to run them.
