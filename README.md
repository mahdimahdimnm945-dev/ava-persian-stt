# Ava — a small, fast Persian speech-to-text model

Ava is a small Persian (Farsi) speech recognition model: just under a
million parameters, trained from scratch on the Neyshekar dataset. It
runs on a phone or a laptop CPU, offline, with no server involved.
Short utterances come back in tens of milliseconds — fast enough to
feel instant. The name is Ava (آوا), the Persian word for voice.

## Who this is for

If you need a small speech-to-text model that turns speech into text on
the spot — on a phone, a laptop, or a small board, with no server and no
internet — and the speech is Persian, this is it. Ava runs on an
ordinary CPU with no GPU, needs about 5 MB of disk, and transcribes a
short utterance in tens of milliseconds.

Persian version of this page: [README-fa.md](README-fa.md)

## What is in here

| Path | What it is |
|---|---|
| `models/stt_v6.onnx` | The model to actually use. Int8 ONNX, 4.8 MB. Takes a raw 16 kHz waveform and returns CTC logits. The mel frontend and the normalization are inside the graph. |
| `models/best.pt` | The PyTorch training checkpoint, 12 MB. |
| `models/stt_v6_int8.tflite` | Full-int8 TFLite, 1.2 MB. Takes log-mel features, shape (1, 40, 600). |
| `model.py` | The network architecture, nothing else. |
| `demo/` | Runnable examples: transcribe a wav file, and a test that runs the model over the included samples. |
| `training/` | The scripts and the guide used to train this model. |

## How good is it

Measured on the Neyshekar validation set (1,884 utterances), character
error rate:

| Model | CER |
|---|---|
| PyTorch, fp32 | 0.2408 |
| ONNX, int8 | 0.2405 |
| TFLite, int8, utterances up to 6 s (1,091 items) | 0.2563 |
| TFLite, int8, all 1,884 items | 0.3492 |

The last row needs a note: the TFLite model has a fixed 6-second input,
so it truncates the 793 longer validation utterances, and that alone is
most of the gap. Quantization itself costs almost nothing here (about
+0.003 on the short subset).

A few more numbers: 996,703 parameters. The ONNX file is 4,798,179 bytes,
the TFLite file 1,183,136 bytes. In our Android test app, one utterance
took 32–62 ms on a POCO M5 phone.

## Try it

```bash
git clone https://github.com/mahdimahdimnm945-dev/ava-persian-stt.git
cd ava-persian-stt/demo
pip install numpy onnxruntime scipy
python test_samples.py
python transcribe.py samples/sample1.wav
```

The smallest possible use, if you just want to call the model from your
own code:

```python
import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

sess = ort.InferenceSession("models/stt_v6.onnx", providers=["CPUExecutionProvider"])
sr, data = wavfile.read("my-recording.wav")
samples = data.astype(np.float32) / 32768.0
logits = sess.run(["logits"], {"waveform": samples})[0]
```

The full pipeline — resampling, the frame count, the CTC collapse — is in
`demo/transcribe.py`. Copy it, it is short.

## What it actually sounds like

Real outputs from the three samples in `demo/samples/`, run with
`python test_samples.py`. Nothing here is cherry-picked to look perfect;
the third one shows the typical failure.

| Sample | Reference | What Ava produced | CER |
|---|---|---|---|
| sample1.wav (2.7 s) | این متخصص می‌گوید | این متخص می‌گوی‌د | 0.118 |
| sample2.wav (3.9 s) | جمع بندی‌ها و کنکورهای سال پیش | جمعوندی‌ها و کنکورهای سال پیش | 0.067 |
| sample3.wav (4.7 s) | برای مثال گردش مالی خودش باشه نه فرد دیگه‌ایی. | برای نسال گردش مالی وش باش نفرد دیگ‌ای | 0.217 |

The model outputs Persian letters, spaces, the ZWNJ character and Persian
digits only. There is no punctuation in the output, and no Latin letters.

## What it is good for

- Persian voice commands on a device, without sending audio to a cloud.
- Voice interfaces that need an answer right away: a short command is
  transcribed in 32–62 ms on a phone CPU.
- Offline transcription on a phone or a PC.
- Smart-home control, accessibility tools, note-taking apps.
- A starting point for fine-tuning on your own Persian speech data.

## What it is not good at

- It was trained on Neyshekar only: read speech, mostly clean recordings.
  Noisy rooms, strong accents, fast overlapping speech and long
  monologues will be worse.
- Typical errors are dropped consonants, especially on fast speech.
- The TFLite model caps input at 6 seconds. The ONNX model accepts
  longer audio, but the numbers above were measured on short utterances.
- It is too slow for real-time use on an ESP32-S3 with the reference
  TFLM kernels: over 90 seconds per command. That is a kernel and
  hardware limit, not the model's size. Use it on a phone, a PC, or a
  board with a real CPU.

## How it was trained

A depthwise-separable CNN with a CTC head, trained from scratch on
Neyshekar: 62,279 clips, about 99 hours (58,167 for training, 1,884 for
validation). AdamW, learning rate 2e-3, batch 128, up to 150 epochs with
early stopping; the best checkpoint came at epoch 116. The audio frontend
is fixed: 16 kHz, 25 ms window, 10 ms hop, 40 log-mel bands, and a
47-character vocabulary. The full recipe, the data preparation and the
exact commands are in [TRAINING.md](TRAINING.md), and the scripts are in
`training/`.

## License

Free to use, copy, modify and distribute for non-commercial purposes,
with attribution. Commercial use needs written permission from the
authors — open an issue in this repository to ask. See
[LICENSE](LICENSE) for the full text.

## Donate

If this is useful to you and you want to support the work, you can send
TON to:

```
UQAWExC4dxbVOcv6R07YORkfrwPQOBtnD3wNTf7qvg9vUsj_
```

## Credits

- Training data: the Neyshekar dataset
  ([shekar-ai/Neyshekar](https://huggingface.co/datasets/shekar-ai/Neyshekar)).
- Built by Mehdi Azami Moghadam.
