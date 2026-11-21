import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile
from scipy.signal import resample_poly

VOCAB = (
    [""]
    + list("اآبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
    + [" ", "‌"]
    + [chr(0x6F0 + i) for i in range(10)]
    + [""]
)

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "stt_v6.onnx"


def load_wav_16k(path):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    else:
        x = data.astype(np.float32)
    if sr != 16000:
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def transcribe(sess, samples):
    n = samples.shape[0]
    if n < 400:
        return ""
    n_frames = (n - 400) // 160 + 1
    out_len = n_frames // 8
    logits = sess.run(["logits"], {"waveform": samples.astype(np.float32)})[0]
    ids = logits[0, :out_len].argmax(axis=-1).tolist()
    out = []
    prev = -1
    for i in ids:
        if i != prev and i != 0:
            out.append(VOCAB[i])
        prev = i
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="wav file to transcribe (any sample rate, mono or stereo)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    samples = load_wav_16k(args.wav)
    print(transcribe(sess, samples))


if __name__ == "__main__":
    main()
