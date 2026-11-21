# Demo

Two small scripts that run Ava on real audio. No GPU needed.

## Install

```bash
pip install numpy onnxruntime scipy
```

## Run the included samples

```bash
python test_samples.py
```

This transcribes the three wav files in `samples/` and prints the reference
text next to what the model actually produced, with the character error rate
(CER) for each one. The samples are short clips from the Neyshekar validation
set, and the reference texts are the ones that came with the dataset.

## Transcribe your own file

```bash
python transcribe.py path/to/your-file.wav
```

Any wav file works: mono or stereo, any sample rate. It gets converted to
16 kHz mono automatically and the transcription is printed to stdout.
Short, clear, single-speaker recordings work best. The model outputs Persian
letters, spaces, the ZWNJ character and Persian digits only, no punctuation.

By default both scripts use `../models/stt_v6.onnx`. Pass `--model` to point
at a different ONNX file.
