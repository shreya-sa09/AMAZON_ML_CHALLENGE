"""Character-level pair model (MatchPyramid-style), trained from scratch in PyTorch. Country-agnostic: operates on the
normalized (ascii-folded, transliterated) name_core / addr_norm strings of both records."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LN, LA = 40, 64
ALPHA = "abcdefghijklmnopqrstuvwxyz0123456789 "
CMAP = np.zeros(256, dtype=np.uint8)
for i, ch in enumerate(ALPHA):
    CMAP[ord(ch)] = i + 2     # 0 = pad, 1 = other
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def encode(strings, L):
    out = np.zeros((len(strings), L), dtype=np.uint8)
    for i, s in enumerate(strings):
        b = (s or "").encode("ascii", "ignore")[:L]
        if b:
            a = np.frombuffer(b, dtype=np.uint8)
            v = CMAP[a]
            v[v == 0] = 1
            out[i, :len(v)] = v
    return out


class Side(nn.Module):
    def __init__(self, d=48):
        super().__init__()
        self.c1 = nn.Conv1d(d, 64, 3, padding=1)
        self.c2 = nn.Conv1d(64, 64, 3, padding=1)

    def forward(self, e):                      # e: (B, L, d)
        h = F.gelu(self.c1(e.transpose(1, 2)))
        h = h + F.gelu(self.c2(h))
        return h.transpose(1, 2)               # (B, L, 64)


class Pyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 48, 3, padding=1), nn.GELU(), nn.AdaptiveMaxPool2d((4, 4)))

    def forward(self, m):
        return self.net(m).flatten(1)          # (B, 768)


class CharPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(ALPHA) + 2, 48, padding_idx=0)
        self.side_n, self.side_a = Side(), Side()
        self.pyr_n, self.pyr_a = Pyramid(), Pyramid()
        self.head = nn.Sequential(nn.Linear(768 * 2 + 4, 128), nn.GELU(), nn.Linear(128, 1))

    def matrix(self, side, qa, sa):
        eq, es = self.emb(qa), self.emb(sa)
        hq, hs = side(eq), side(es)
        sim = torch.bmm(hq, hs.transpose(1, 2)) / 8.0
        exact = ((qa[:, :, None] == sa[:, None, :]) & (qa[:, :, None] > 0)).float()
        mask = ((qa[:, :, None] > 0) & (sa[:, None, :] > 0)).float()
        return torch.stack([sim * mask, exact, mask], 1)

    def forward(self, qn, sn, qa, sa):
        mn, ma = self.matrix(self.side_n, qn, sn), self.matrix(self.side_a, qa, sa)
        lens = torch.stack([(qn > 0).sum(1), (sn > 0).sum(1), (qa > 0).sum(1), (sa > 0).sum(1)], 1).float() / 64.0
        return self.head(torch.cat([self.pyr_n(mn), self.pyr_a(ma), lens], 1)).squeeze(1)


def train_model(X, y, epochs=2, bs=2048, lr=2e-3, seed=42, log=print):
    """X: dict of uint8 arrays qn, sn, qa, sa (N, L). Returns trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = CharPair().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    Xg = {k: torch.from_numpy(v).to(DEV) for k, v in X.items()}
    yg = torch.from_numpy(y.astype(np.float32)).to(DEV)
    n = len(y)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            with torch.autocast("cuda", dtype=torch.float16):
                logit = m(Xg["qn"][idx].long(), Xg["sn"][idx].long(), Xg["qa"][idx].long(), Xg["sa"][idx].long())
                loss = F.binary_cross_entropy_with_logits(logit.float(), yg[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(idx)
            step += 1
            if step % 200 == 0:
                log(f"  epoch {ep + 1} step {step}/{steps} loss {tot / min(n, s + bs):.4f}")
        log(f"epoch {ep + 1} mean loss {tot / n:.4f}")
    del Xg, yg
    return m


@torch.no_grad()
def predict(m, X, bs=8192):
    m.eval()
    out = []
    n = len(X["qn"])
    for s in range(0, n, bs):
        b = {k: torch.from_numpy(v[s:s + bs]).to(DEV).long() for k, v in X.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(m(b["qn"], b["sn"], b["qa"], b["sa"]).float()).cpu().numpy())
    m.train()
    return np.concatenate(out)
