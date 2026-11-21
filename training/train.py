import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


LETTERS = list("اآبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
VOCAB = (
    ["<blank>"] + LETTERS + [" ", "‌"] + [chr(0x6F0 + _i) for _i in range(10)] + ["<unk>"]
)
VOCAB_SIZE = len(VOCAB)
TIME_STRIDE = 8
INT8_OVERHEAD_BYTES = 4096


class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = torch.relu(self.bn1(self.dw(x)))
        x = torch.relu(self.bn2(self.pw(x)))
        return x


class DSCNNv6(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 128, 3, stride=(2, 2), padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.blocks = nn.Sequential(
            DSBlock(128, 192),
            DSBlock(192, 192, stride=(2, 2)),
            DSBlock(192, 256),
            DSBlock(256, 256, stride=(1, 2)),
            DSBlock(256, 336),
            DSBlock(336, 336),
            DSBlock(336, 336),
            DSBlock(336, 336),
            DSBlock(336, 336),
            DSBlock(336, 336),
            DSBlock(336, 336),
        )
        self.head = nn.Linear(336, vocab)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn0(self.conv0(x)))
        x = self.blocks(x)
        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def edit_distance(a, b):
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
        prev = cur
    return prev[lb]


def collapse_ids(ids):
    out, prev = [], -1
    for i in ids:
        if i != prev and i != 0:
            out.append(i)
        prev = i
    return out


def ctc_min_frames(tokens):
    reps = 0
    for a, b in zip(tokens, tokens[1:]):
        if a == b:
            reps += 1
    return len(tokens) + reps


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_batch(items, idxs, device, mean, std, augment=False):
    sel = [items[j] for j in idxs]
    tmax = max(it["feat"].shape[0] for it in sel)
    x = torch.zeros(len(sel), tmax, 40)
    lens = []
    for k, it in enumerate(sel):
        f = torch.tensor(it["feat"].astype(np.float32))
        f = (f - mean) / std
        if augment:
            for _ in range(2):
                fw = int(np.random.randint(0, 9))
                f0 = int(np.random.randint(0, max(1, 40 - fw)))
                f[:, f0 : f0 + fw] = 0.0
            for _ in range(2):
                tw = int(np.random.randint(0, 41))
                t0 = int(np.random.randint(0, max(1, f.shape[0] - tw)))
                f[t0 : t0 + tw, :] = 0.0
        x[k, : f.shape[0]] = f
        lens.append(f.shape[0])
    y = [torch.tensor(it["tokens"], dtype=torch.long) for it in sel]
    return x.permute(0, 2, 1).to(device), torch.tensor(lens), y


def bucketed_permutation(lengths, batch, rng):
    order = sorted(range(len(lengths)), key=lambda j: lengths[j])
    buckets = [order[i : i + batch * 50] for i in range(0, len(order), batch * 50)]
    rng.shuffle(buckets)
    perm = []
    for b in buckets:
        b = list(b)
        rng.shuffle(b)
        perm.extend(b)
    return np.array(perm, dtype=np.int64)


def cer_of(model, items, device, mean, std, limit, amp):
    model.eval()
    tot_ed, tot_ch = 0, 0
    with torch.no_grad():
        for s in range(0, min(limit, len(items)), 32):
            idxs = list(range(s, min(s + 32, len(items))))
            x, lens, _ = make_batch(items, idxs, device, mean, std)
            if amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(x)
            else:
                logits = model(x)
            logits = logits.float().cpu()
            outs = (lens // TIME_STRIDE).tolist()
            for k, j in enumerate(idxs):
                hyp_ids = logits[k, : outs[k]].argmax(dim=-1).tolist()
                hyp = collapse_ids(hyp_ids)
                tot_ed += edit_distance(hyp, items[j]["tokens"])
                tot_ch += max(1, len(items[j]["tokens"]))
    return tot_ed / max(1, tot_ch)


def build_checkpoint(model, opt, sched, epoch, best_cer, patience, rng, config_fp):
    ckpt = {
        "version": 6,
        "config_fp": config_fp,
        "state": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": epoch,
        "best_cer": best_cer,
        "patience": patience,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        "rng_gen": rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        ckpt["rng_cuda"] = torch.cuda.get_rng_state_all()
    return ckpt


def restore_rng(ckpt, rng):
    torch.set_rng_state(ckpt["rng_torch"])
    np.random.set_state(ckpt["rng_numpy"])
    random.setstate(ckpt["rng_python"])
    rng.bit_generator.state = ckpt["rng_gen"]
    if "rng_cuda" in ckpt and ckpt["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])


def prune_periodic(out_dir, keep_last):
    if keep_last <= 0:
        return
    files = sorted(out_dir.glob("ckpt_ep*.pt"))
    for path in files[: max(0, len(files) - keep_last)]:
        path.unlink()


def atomic_torch_save(obj, path):
    tmp = path.with_suffix(".tmp.pt")
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def hf_client(repo_id, token):
    try:
        from huggingface_hub import HfApi, create_repo

        create_repo(repo_id, token=token, private=True, exist_ok=True)
        return HfApi(token=token)
    except Exception as e:
        print("hf repo setup failed:", e)
        return None


def hf_upload(api, repo_id, local_path, repo_path):
    try:
        api.upload_file(
            path_or_fileobj=str(local_path), path_in_repo=repo_path, repo_id=repo_id
        )
        print("hf uploaded", repo_path)
        return True
    except Exception as e:
        print("hf upload failed:", repo_path, e)
        return False


def hf_write_state(out_dir, epoch, best_cer):
    state_path = out_dir / "hf_state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(
            {"epoch": epoch, "best_cer": round(best_cer, 4), "updated": time.time()}, f
        )
    return state_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="local_prep/features")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--dev-limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--out", default="ckpt_v6")
    p.add_argument("--resume", default=None)
    p.add_argument("--hf-repo", default="")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--zero-infinity", action="store_true")
    p.add_argument("--bucket", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--specaug", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    feat_dir = Path(args.features)
    stats = json.loads((feat_dir / "stats.json").read_text())
    mean = torch.tensor(stats["mean"])
    std = torch.tensor(stats["std"]).clamp(min=1e-3)
    train = load_pickle(feat_dir / "real_train.pkl")
    dev = load_pickle(feat_dir / "real_dev.pkl")
    src_counts = Counter(it["src"] for it in train)
    print("train src counts:", dict(src_counts))
    train = [it for it in train if it["src"] in ("fleurs_train", "neyshekar_train")]
    dropped = sum(v for k, v in src_counts.items() if k not in ("fleurs_train", "neyshekar_train"))
    if dropped:
        print("WARNING: dropped", dropped, "train items with unexpected src")
    alignable = [it for it in train if ctc_min_frames(it["tokens"]) <= it["feat"].shape[0] // TIME_STRIDE]
    dropped_align = len(train) - len(alignable)
    train = alignable
    if dropped_align:
        print("dropped", dropped_align, "train items unalignable for CTC (tokens + repeats > T//8 outputs)")
    dev_bad = sum(1 for it in dev if ctc_min_frames(it["tokens"]) > it["feat"].shape[0] // TIME_STRIDE)
    if dev_bad:
        print("note:", dev_bad, "dev items exceed T//8 outputs (kept for eval comparability)")

    epochs = args.epochs
    batch = args.batch
    dev_limit = args.dev_limit if args.dev_limit > 0 else len(dev)
    ckpt_every = args.ckpt_every
    import hashlib

    config_fp = hashlib.sha256(
        json.dumps(
            {
                "batch": batch,
                "lr": args.lr,
                "lr_min": args.lr_min,
                "wd": args.wd,
                "clip": args.clip,
                "warmup": args.warmup,
                "bucket": args.bucket,
                "specaug": args.specaug,
                "zero_infinity": args.zero_infinity,
                "amp": args.amp,
                "seed": args.seed,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    run_t0 = time.time()
    if args.smoke:
        train = train[:300]
        dev = dev[:200]
        epochs = 2
        batch = 8
        dev_limit = 50
        ckpt_every = 1
        print("SMOKE MODE: 300 train items, 200 dev items, 2 epochs, batch 8")

    model = DSCNNv6(VOCAB_SIZE).to(device)
    nparams = count_params(model)
    int8_bytes = nparams + INT8_OVERHEAD_BYTES
    print("device:", device)
    print("vocab size:", VOCAB_SIZE)
    print("parameters:", nparams)
    print("int8 weight size estimate:", int8_bytes, "bytes =", round(int8_bytes / 1048576, 3), "MiB")
    print("train items:", len(train), "dev items:", len(dev))
    print(
        "epochs:", epochs, "batch:", batch, "lr:", args.lr,
        "warmup:", args.warmup, "patience:", args.patience,
        "ckpt every:", ckpt_every, "amp:", args.amp,
        "bucket:", args.bucket, "specaug:", args.specaug,
        "max hours:", args.max_hours, "config:", config_fp,
    )
    hf_api = None
    hf_repo = args.hf_repo
    if hf_repo:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            hf_api = hf_client(hf_repo, hf_token)
            if hf_api is not None:
                print("hf backup enabled:", hf_repo)
        else:
            print("hf backup disabled: HF_TOKEN not set")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup = args.warmup
    floor_frac = args.lr_min / args.lr

    def lr_factor(epoch):
        if epoch < warmup:
            return 0.1 + 0.9 * epoch / max(1, warmup)
        span = max(1, epochs - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup) / span))
        return floor_frac + (1.0 - floor_frac) * cos

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    ctc = nn.CTCLoss(blank=0, zero_infinity=args.zero_infinity)

    start_epoch = 0
    best_cer = 1e9
    patience = 0
    completed = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt.get("config_fp") and ckpt["config_fp"] != config_fp:
            print(
                "WARNING: checkpoint config", ckpt["config_fp"],
                "differs from current config", config_fp,
            )
        if "optimizer" in ckpt and "scheduler" in ckpt:
            model.load_state_dict(ckpt["state"])
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"]
            completed = start_epoch
            best_cer = ckpt["best_cer"]
            patience = ckpt.get("patience", 0)
            if "rng_torch" in ckpt:
                restore_rng(ckpt, rng)
            print("resumed from", args.resume, "at epoch", start_epoch, "best CER", round(best_cer, 4))
        else:
            model.load_state_dict(ckpt["state"])
            best_cer = ckpt.get("cer", 1e9)
            print("loaded legacy weights-only checkpoint", args.resume, "best CER", round(best_cer, 4))
        if log_path.exists():
            kept = []
            for line in log_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("epoch", 0) <= start_epoch:
                    kept.append(line)
            log_path.write_text("\n".join(kept) + ("\n" if kept else ""))
        probe_x, _, _ = make_batch(dev[:2], [0, 1], device, mean, std)
        model.eval()
        with torch.no_grad():
            probe_out = model(probe_x)
        model.train()
        print("resume forward check ok:", tuple(probe_out.shape))

    log_f = open(log_path, "a", encoding="utf-8")
    status = 0
    lengths = [int(it["feat"].shape[0]) for it in train]
    try:
        for ep in range(start_epoch, epochs):
            if args.max_hours > 0 and (time.time() - run_t0) / 3600.0 >= args.max_hours:
                print("time budget reached (%.1f h); stopping cleanly at epoch %d" % (args.max_hours, ep))
                (out_dir / "STOPPED").write_text(
                    "time budget %.1f h reached at epoch %d\n" % (args.max_hours, ep)
                )
                status = 3
                break
            model.train()
            if args.bucket:
                perm = bucketed_permutation(lengths, batch, rng)
            else:
                perm = rng.permutation(len(train))
            tot_loss, nb = 0.0, 0
            t0 = time.time()
            nan_hit = None
            for s in range(0, len(perm), batch):
                idxs = perm[s : s + batch].tolist()
                x, lens, y = make_batch(train, idxs, device, mean, std, augment=args.specaug)
                if args.amp:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        logits = model(x)
                else:
                    logits = model(x)
                lp = logits.float().log_softmax(dim=-1).permute(1, 0, 2).cpu()
                loss = ctc(
                    lp,
                    torch.cat(y),
                    lens // TIME_STRIDE,
                    torch.tensor([len(t) for t in y]),
                )
                if not torch.isfinite(loss):
                    nan_hit = "loss"
                    break
                opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                if not math.isfinite(float(grad_norm)):
                    nan_hit = "grad_norm"
                    break
                opt.step()
                tot_loss += loss.item()
                nb += 1
                if nb % 100 == 0:
                    log_f.write(
                        json.dumps(
                            {
                                "kind": "step",
                                "epoch": ep + 1,
                                "step": nb,
                                "loss": round(loss.item(), 4),
                                "grad_norm": round(float(grad_norm), 3),
                            }
                        )
                        + "\n"
                    )
                    log_f.flush()
            if nan_hit:
                bad_path = out_dir / ("ckpt_nan_ep%04d.pt" % (ep + 1))
                atomic_torch_save(
                    build_checkpoint(model, opt, sched, ep, best_cer, patience, rng, config_fp),
                    bad_path,
                )
                print("ABORT: non-finite", nan_hit, "at epoch", ep + 1)
                print("emergency checkpoint saved to", bad_path)
                status = 2
                break
            sched.step()
            avg_loss = tot_loss / max(nb, 1)
            cer = cer_of(model, dev, device, mean, std, dev_limit, args.amp)
            seconds = time.time() - t0
            lr_now = opt.param_groups[0]["lr"]
            improved = cer < best_cer
            if improved:
                best_cer = cer
                patience = 0
            else:
                patience += 1
            rec = {
                "kind": "epoch",
                "epoch": ep + 1,
                "loss": round(avg_loss, 4),
                "lr": lr_now,
                "dev_cer": round(cer, 4),
                "best_cer": round(best_cer, 4),
                "seconds": round(seconds, 1),
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(
                "epoch %d/%d loss %.4f lr %.2e dev CER %.4f best %.4f %.1f s"
                % (ep + 1, epochs, avg_loss, lr_now, cer, best_cer, seconds)
            )
            ckpt = build_checkpoint(model, opt, sched, ep + 1, best_cer, patience, rng, config_fp)
            completed = ep + 1
            atomic_torch_save(ckpt, out_dir / "last.pt")
            if improved:
                atomic_torch_save(ckpt, out_dir / "best.pt")
                if hf_api is not None:
                    hf_upload(hf_api, hf_repo, out_dir / "best.pt", "best.pt")
                    hf_upload(
                        hf_api,
                        hf_repo,
                        hf_write_state(out_dir, ep + 1, best_cer),
                        "training_state.json",
                    )
            if (ep + 1) % ckpt_every == 0 or ep + 1 == epochs:
                atomic_torch_save(ckpt, out_dir / ("ckpt_ep%04d.pt" % (ep + 1)))
                prune_periodic(out_dir, args.keep_last)
            if patience >= args.patience:
                print("early stop: patience", args.patience, "reached at epoch", ep + 1)
                break
    except KeyboardInterrupt:
        atomic_torch_save(
            build_checkpoint(model, opt, sched, completed, best_cer, patience, rng, config_fp),
            out_dir / "last.pt",
        )
        print("interrupted: checkpoint saved to", out_dir / "last.pt", "at epoch", completed)
        status = 130
    finally:
        log_f.close()
    if hf_api is not None:
        if (out_dir / "best.pt").exists():
            hf_upload(hf_api, hf_repo, out_dir / "best.pt", "best.pt")
        if (out_dir / "last.pt").exists():
            hf_upload(hf_api, hf_repo, out_dir / "last.pt", "last.pt")
        if log_path.exists():
            hf_upload(hf_api, hf_repo, log_path, "train_log.jsonl")
        hf_upload(
            hf_api, hf_repo, hf_write_state(out_dir, completed, best_cer), "training_state.json"
        )
    print("done. best dev CER:", round(best_cer, 4))
    return status


if __name__ == "__main__":
    sys.exit(main())
