# train_stageB_3text_multitask.py
import argparse, os, re, math, random, warnings
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    CLIPModel, CLIPProcessor, AutoTokenizer, get_cosine_schedule_with_warmup
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# -------------------- utils --------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def clean_text(s, max_chars=480):
    if s is None or (isinstance(s,float) and np.isnan(s)): return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:max_chars]

def to_float(x):
    try: return float(x)
    except: return np.nan

def format_budget_as_text(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "Budget: unknown"
    s = str(x).strip()
    val = None
    try:
        val = float(s.replace(",","").replace("$",""))
    except Exception:
        m = re.match(r"^\$?\s*([0-9]*\.?[0-9]+)\s*([MmBb])$", s)
        if m:
            val = float(m.group(1))
            val *= 1e6 if m.group(2).upper()=="M" else 1e9
    return f"Budget: ${int(val):,}" if val is not None else f"Budget: {clean_text(s,48)}"

# token-aware trimming for CLIP (77-token window)
_CLIP_TEXT_MAX = 77
_tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def trim_to_tokens(text: str, max_tokens=60) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    ids = _tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return _tok.decode(ids[:max_tokens], skip_special_tokens=True)

def make_bins(y_log, n_bins=10):
    q = np.quantile(y_log, np.linspace(0,1,n_bins+1))
    for i in range(1,len(q)):
        if q[i] <= q[i-1]: q[i] = q[i-1] + 1e-6
    return q

def digitize_bins(y_log, edges):
    return np.clip(np.searchsorted(edges, y_log, side="right")-1, 0, len(edges)-2)

# -------------------- dataset --------------------
REQUIRED = ["Series_Title","meta_prompt","overview","Poster_Image_Path","Gross","budget"]

class MovieDataset(Dataset):
    """
    Inputs:
      - image: Poster_Image_Path
      - text: meta_prompt (trimmed), overview (trimmed), budget_text (generated)
      - target: log1p(Gross)
      - optional: y_bin (set after make_bins on train)
    """
    def __init__(self, csv_path, bin_edges=None):
        df = pd.read_csv(csv_path)
        miss = [c for c in REQUIRED if c not in df.columns]
        if miss: raise ValueError(f"Missing columns in {csv_path}: {miss}")

        df["Gross"] = df["Gross"].apply(to_float)
        df = df.dropna(subset=["Poster_Image_Path","Gross"]).reset_index(drop=True)

        # token-aware trimming to fit CLIP's 77-token limit comfortably
        df["meta_prompt"] = df["meta_prompt"].apply(lambda s: trim_to_tokens(clean_text(s), 60))
        df["overview"]    = df["overview"].apply(lambda s: trim_to_tokens(clean_text(s), 60))
        df["budget_text"] = df["budget"].apply(format_budget_as_text)

        df["y_log1p"] = np.log1p(df["Gross"].astype(float))
        self.df = df

        self.bin_edges = bin_edges
        if bin_edges is not None:
            self.df["y_bin"] = digitize_bins(self.df["y_log1p"].values, bin_edges)

    def set_bins(self, bin_edges):
        self.bin_edges = bin_edges
        self.df["y_bin"] = digitize_bins(self.df["y_log1p"].values, bin_edges)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        try:
            img = Image.open(r["Poster_Image_Path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB",(224,224),(127,127,127))
        return {
            "image": img,
            "meta_prompt": r["meta_prompt"],
            "overview": r["overview"],
            "budget_text": r["budget_text"],
            "y_log1p": torch.tensor(r["y_log1p"], dtype=torch.float32),
            "y_bin": torch.tensor(int(r["y_bin"]) if "y_bin" in r else -1, dtype=torch.long),
        }

def collate_keep_pil(batch):
    return {
        "image":        [b["image"] for b in batch],
        "meta_prompt":  [b["meta_prompt"] for b in batch],
        "overview":     [b["overview"] for b in batch],
        "budget_text":  [b["budget_text"] for b in batch],
        "y_log1p":      torch.stack([b["y_log1p"] for b in batch], 0),
        "y_bin":        torch.stack([b["y_bin"] for b in batch], 0),
    }

# -------------------- head --------------------
class FusionMultiTaskHead(nn.Module):
    """
    Fusion over [image, t_meta, t_over, t_budget] -> shared tower
    Heads:
      - regression (standardized log target)
      - classification over K bins
    """
    def __init__(self, emb_dim, n_bins, drop=0.4):
        super().__init__()
        # start with a slight preference for text
        self.w = nn.Parameter(torch.tensor([0.15, 0.5, 0.3, 0.05], dtype=torch.float32))
        self.shared = nn.Sequential(
            nn.Linear(emb_dim, 2048), nn.GELU(), nn.Dropout(drop),
            nn.Linear(2048, 512), nn.GELU(), nn.Dropout(drop)
        )
        self.reg = nn.Linear(512, 1)
        self.cls = nn.Linear(512, n_bins)

        self.out_scale = nn.Parameter(torch.tensor(1.0))  # sigma
        self.out_bias  = nn.Parameter(torch.tensor(0.0))  # mu

    def set_output_scaling(self, mu, sigma):
        with torch.no_grad():
            self.out_scale.copy_(torch.tensor(float(sigma)))
            self.out_bias.copy_(torch.tensor(float(mu)))

    def forward(self, v, t_meta, t_over, t_budget, return_std=False):
        w = torch.softmax(self.w, dim=0)
        z = nn.functional.normalize(w[0]*v + w[1]*t_meta + w[2]*t_over + w[3]*t_budget, dim=-1)
        h = self.shared(z)
        y_std = self.reg(h).squeeze(-1)                 # standardized log target
        y_log = y_std * self.out_scale + self.out_bias  # de-standardized log
        logits = self.cls(h)                            # class bins
        return (y_log, y_std, logits) if return_std else (y_log, logits)

# -------------------- encode --------------------
@torch.no_grad()
def encode_batch(clip, processor, images, meta, over, budg, device):
    max_len = _CLIP_TEXT_MAX  # hard cap for CLIP
    img_inputs = processor.image_processor(images=images, return_tensors="pt")
    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
    v = nn.functional.normalize(clip.get_image_features(**img_inputs), dim=-1)

    def enc(texts):
        toks = processor.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        t = clip.get_text_features(**toks)
        return nn.functional.normalize(t, dim=-1)

    return v, enc(meta), enc(over), enc(budg)

# -------------------- eval --------------------
@torch.no_grad()
def evaluate(loader, clip, processor, head, device):
    preds, gts = [], []
    for b in loader:
        v, t1, t2, t3 = encode_batch(clip, processor, b["image"], b["meta_prompt"], b["overview"], b["budget_text"], device)
        y_log, logits = head(v, t1, t2, t3)
        preds.append(y_log.cpu().numpy()); gts.append(b["y_log1p"].cpu().numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    pr = np.expm1(preds); gt = np.expm1(gts)
    mae   = mean_absolute_error(gt, pr)
    medae = median_absolute_error(gt, pr)
    rmse  = math.sqrt(mean_squared_error(gt, pr))
    smape = np.mean(2*np.abs(pr-gt)/(np.abs(pr)+np.abs(gt)+1e-6))
    r2    = r2_score(gt, pr)
    mask  = gt >= 1e6
    fmape = np.mean(np.abs((gt[mask]-pr[mask])/gt[mask])) if mask.sum()>0 else np.nan
    return mae, medae, rmse, fmape, smape, r2

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_enc",  type=float, default=3e-6)  # small LR for partial unfreeze
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--gpu",    type=int, default=1)
    ap.add_argument("--n_bins", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # datasets
    train_ds = MovieDataset(args.train)
    val_ds   = MovieDataset(args.val)

    # target scaling on TRAIN ONLY
    y_log_tr = train_ds.df["y_log1p"].values.astype(np.float32)
    mu_y, sd_y = float(y_log_tr.mean()), float(y_log_tr.std()+1e-8)

    # bins on TRAIN ONLY, then set for both
    bin_edges = make_bins(y_log_tr, n_bins=args.n_bins)
    train_ds.set_bins(bin_edges); val_ds.set_bins(bin_edges)

    # class-balanced sampler to avoid mid-range dominance
    counts = train_ds.df["y_bin"].value_counts().sort_index().values
    inv_freq = 1.0 / (counts + 1e-6)
    weights = inv_freq[train_ds.df["y_bin"].values]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                          num_workers=4, pin_memory=True, collate_fn=collate_keep_pil)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=collate_keep_pil)

    # model
    clip = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    head = FusionMultiTaskHead(emb_dim=clip.config.projection_dim, n_bins=args.n_bins, drop=0.4).to(device)
    head.set_output_scaling(mu_y, sd_y)

    # deeper partial unfreeze (last 4 transformer blocks + projections)
    for p in clip.parameters(): p.requires_grad = False
    try:
        for blk in clip.vision_model.encoder.layers[-4:]:
            for p in blk.parameters(): p.requires_grad = True
    except Exception: pass
    try:
        for blk in clip.text_model.encoder.layers[-4:]:
            for p in blk.parameters(): p.requires_grad = True
    except Exception: pass
    for p in clip.visual_projection.parameters(): p.requires_grad = True
    for p in clip.text_projection.parameters():   p.requires_grad = True

    def bn_eval(m):
        if m.__class__.__name__ in ["BatchNorm2d","FrozenBatchNorm2d"]:
            m.eval()
    if hasattr(clip, "vision_model"): clip.vision_model.apply(bn_eval)

    # opt/sched/loss
    enc_params = [p for p in clip.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": args.lr_head, "weight_decay": 0.05},
         {"params": enc_params,        "lr": args.lr_enc,  "weight_decay": 0.01}],
        betas=(0.9,0.98), eps=1e-6
    )
    total_steps = max(1, len(train_dl)*args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05*total_steps), total_steps)

    huber = nn.HuberLoss(delta=1.0)
    ce    = nn.CrossEntropyLoss(label_smoothing=0.05)

    scaler = torch.amp.GradScaler('cuda') if device.startswith("cuda") else None
    amp_ctx = (lambda: torch.amp.autocast('cuda')) if device.startswith("cuda") else nullcontext

    best_mae = float("inf")
    patience, since_best = 8, 0  # early stop on MAE
    for epoch in range(1, args.epochs+1):
        head.train(); clip.train()
        tr_losses = []
        for b in train_dl:
            y_log = b["y_log1p"].to(device)
            y_std = (y_log - mu_y) / sd_y
            y_bin = b["y_bin"].to(device)

            with amp_ctx():
                v, t1, t2, t3 = encode_batch(clip, processor, b["image"], b["meta_prompt"], b["overview"], b["budget_text"], device)
                y_log_pred, y_std_pred, logits = head(v, t1, t2, t3, return_std=True)

                loss_reg = huber(y_std_pred, y_std)
                loss_cls = ce(logits, y_bin)
                loss = loss_reg + 0.5 * loss_cls

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            tr_losses.append(loss.item())

        head.eval(); clip.eval()
        mae, medae, rmse, fmape, smape, r2 = evaluate(val_dl, clip, processor, head, device)
        print(f"[Stage-B(3text+MTL) {epoch}/{args.epochs}] "
              f"train_loss={np.mean(tr_losses):.4f}  "
              f"val_MAE={mae:,.2f}  val_MedAE={medae:,.2f}  val_RMSE={rmse:,.2f}  "
              f"val_fMAPE(>=$1M)={(100*fmape if fmape==fmape else float('nan')):.2f}%  "
              f"val_sMAPE={100*smape:.2f}%  val_R2={r2:.3f}")

        improved = mae < best_mae - 1e2  # require tiny improvement to reset patience
        if improved:
            best_mae = mae; since_best = 0
            torch.save({
                "head": head.state_dict(), "clip": clip.state_dict(),
                "mu_y": mu_y, "sd_y": sd_y, "bin_edges": bin_edges
            }, "model_stageB_3text_multitask.pt")
            print("  ↳ saved model_stageB_3text_multitask.pt (best MAE)")
        else:
            since_best += 1
            if since_best >= patience:
                print("Early stopping: no MAE improvement.")
                break

if __name__ == "__main__":
    main()