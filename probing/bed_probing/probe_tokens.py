"""Train T5 ridge, attention and token-sum readouts on extracted frozen tokens. See README.md."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV

logger = logging.getLogger("probe_tokens")


# ── data ───────────────────────────────────────────────────────────────────────

def load(alias: str, tokens_dir: Path) -> dict:
    return {s: torch.load(tokens_dir / alias / f"{s}.pt") for s in ("train", "val", "test")}


def pad(objs: list, mu: torch.Tensor, sd: torch.Tensor):
    """(view1, view2) token lists -> X [n, N, D] standardized float16, view id [n, N], mask [n, N] (True = pad)."""
    lens = [a.shape[0] + b.shape[0] for a, b in objs]
    n, N, D = len(objs), max(lens), objs[0][0].shape[1]
    X = torch.zeros(n, N, D, dtype=torch.float16)
    view = torch.zeros(n, N, dtype=torch.long)
    mask = torch.ones(n, N, dtype=torch.bool)
    for i, (a, b) in enumerate(objs):
        t = torch.cat([a, b]).float()
        X[i, : len(t)] = ((t - mu) / sd).half()
        view[i, a.shape[0]: len(t)] = 1
        mask[i, : len(t)] = False
    return X, view, mask


# ── probes ─────────────────────────────────────────────────────────────────────

class AttnProbe(nn.Module):
    def __init__(self, d: int, h: int = 256, k: int = 8, out: int = 12, p: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p))
        self.view = nn.Embedding(2, h)
        self.q = nn.Parameter(torch.randn(k, h) * 0.02)
        self.attn = nn.MultiheadAttention(h, 4, dropout=p, batch_first=True)
        self.head = nn.Sequential(nn.Linear(k * h, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, out))

    def forward(self, x, view, mask):
        z = self.proj(x) + self.view(view)
        o, _ = self.attn(self.q.expand(len(x), -1, -1), z, z, key_padding_mask=mask)
        return F.softplus(self.head(o.flatten(1)))


class TokenSumProbe(nn.Module):
    def __init__(self, d: int, h: int = 256, out: int = 12, p: float = 0.1) -> None:
        super().__init__()
        self.view = nn.Embedding(2, d)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, out))
        # Start near zero per token: with ~1,350 tokens a default init sums to ~1,000 per color, the first
        # updates push every unit into softplus' flat region and the probe collapses to all-zero counts.
        nn.init.constant_(self.mlp[-1].bias, -7.0)

    def forward(self, x, view, mask):
        s = F.softplus(self.mlp(x + self.view(view)))
        return (s * (~mask).unsqueeze(-1)).sum(1)


def train_neural(cls, data: dict, seed: int, device: str, epochs: int = 300, patience: int = 30) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, vtr, mtr, ytr = data["train"]
    model = cls(Xtr.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def predict(split):
        X, v, m, _ = data[split]
        model.eval()
        with torch.no_grad():
            return torch.cat([model(X[i:i + 50].to(device).float(), v[i:i + 50].to(device), m[i:i + 50].to(device)).cpu()
                              for i in range(0, len(X), 50)])

    best, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            loss = F.smooth_l1_loss(model(Xtr[b].to(device).float(), vtr[b].to(device), mtr[b].to(device)),
                                    ytr[b].to(device).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = (predict("val") - data["val"][3].float()).abs().mean().item()
        if val < best - 1e-4:
            best, wait = val, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    logger.info("  %s seed %d: stopped at epoch %d, best val MAE %.3f", cls.__name__, seed, ep + 1, best)
    return predict("test").numpy()


# ── metrics (same derivations as the VLM analyses) ─────────────────────────────

def dominant(v: np.ndarray) -> frozenset:
    return frozenset(np.flatnonzero(v == v.max())) if v.max() > 0 else frozenset()


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    r = np.clip(np.round(pred), 0, None)
    present = y > 0
    return {
        "per_color_mae": float(np.abs(pred - y).mean()),
        "T5_exact": 100 * float((r == y).all(1).mean()),
        "present_color_exact": 100 * float((r[present] == y[present]).mean()),
        "T1_from_counts": 100 * float((r.sum(1) == y.sum(1)).mean()),
        "T4_from_counts": 100 * float(((r > 0).sum(1) == present.sum(1)).mean()),
        "T7_from_counts": 100 * float(np.mean([dominant(a) == dominant(b) for a, b in zip(r, y)])),
    }


def summarize(runs: list) -> dict:
    return {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))] for k in runs[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu", help="cpu (default), mps or cuda")
    ap.add_argument("--probes", nargs="+", default=["mean_ridge", "attn", "token_sum"])
    ap.add_argument("--n-train", type=int, default=None, help="use only the first N training objects (learning curve)")
    ap.add_argument("--tokens-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("probe_results"))
    args = ap.parse_args()
    if args.seeds < 1 or (args.n_train is not None and args.n_train < 1):
        ap.error("--seeds and --n-train must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_path = args.out_dir / (f"results_n{args.n_train}.json" if args.n_train else "results.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for alias in args.encoders:
        raw = load(alias, args.tokens_dir)
        if args.n_train and args.n_train > len(raw["train"]["counts"]):
            ap.error("--n-train exceeds available training objects")
        if args.n_train:
            raw["train"] = {k: v[: args.n_train] for k, v in raw["train"].items()}
        y = {s: raw[s]["counts"].numpy() for s in raw}
        res = results.get(alias, {})
        res["constant_mean"] = metrics(np.tile(y["train"].mean(0), (len(y["test"]), 1)), y["test"])

        # mean-pooled ridge (paper readout) on the same split
        if "mean_ridge" in args.probes:
            feats = {s: np.stack([np.concatenate([a.float().mean(0).numpy(), b.float().mean(0).numpy()])
                                  for a, b in raw[s]["tokens"]]) for s in raw}
            ridge = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(np.vstack([feats["train"], feats["val"]]),
                                                               np.vstack([y["train"], y["val"]]))
            res["mean_ridge"] = metrics(ridge.predict(feats["test"]), y["test"])
            logger.info("%s mean_ridge: %s", alias, res["mean_ridge"])

        # token-preserving probes
        sample = torch.cat([torch.cat(v) for v in raw["train"]["tokens"][:200]]).float()
        mu, sd = sample.mean(0), sample.std(0) + 1e-5
        data = {s: (*pad(raw[s]["tokens"], mu, sd), raw[s]["counts"]) for s in raw}
        for name, cls in (("attn", AttnProbe), ("token_sum", TokenSumProbe)):
            if name not in args.probes:
                continue
            runs = [metrics(train_neural(cls, data, seed, args.device), y["test"]) for seed in range(args.seeds)]
            res[name] = summarize(runs)
            logger.info("%s %s: %s", alias, name, res[name])
        results[alias] = res
        out_path.write_text(json.dumps(results, indent=1))

    cols = ["per_color_mae", "T5_exact", "present_color_exact", "T1_from_counts", "T4_from_counts", "T7_from_counts"]
    print("\n" + " | ".join(["encoder", "probe"] + cols))
    for alias, res in results.items():
        for probe, m in res.items():
            vals = [f"{m[c][0]:.2f}±{m[c][1]:.2f}" if isinstance(m[c], list) else f"{m[c]:.2f}" for c in cols]
            print(" | ".join([alias, probe] + vals))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
