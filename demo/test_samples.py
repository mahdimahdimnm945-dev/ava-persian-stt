import argparse
import sys
import time
from pathlib import Path

import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe import DEFAULT_MODEL, load_wav_16k, transcribe

SAMPLES = [
    ("sample1.wav", "این متخصص می‌گوید"),
    ("sample2.wav", "جمع بندی‌ها و کنکورهای سال پیش"),
    ("sample3.wav", "برای مثال گردش مالی خودش باشه نه فرد دیگه‌ایی."),
]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[len(b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    here = Path(__file__).resolve().parent
    tot_ed, tot_ch = 0, 0
    for name, ref in SAMPLES:
        samples = load_wav_16k(str(here / "samples" / name))
        t0 = time.time()
        hyp = transcribe(sess, samples)
        ms = (time.time() - t0) * 1000
        ed = edit_distance(hyp, ref)
        tot_ed += ed
        tot_ch += max(1, len(ref))
        print("---", name)
        print("reference:   ", ref)
        print("transcribed: ", hyp)
        print("CER: %.3f  (%.0f ms)" % (ed / max(1, len(ref)), ms))
    print("overall CER on %d samples: %.3f" % (len(SAMPLES), tot_ed / max(1, tot_ch)))


if __name__ == "__main__":
    main()
