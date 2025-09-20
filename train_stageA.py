# train_stageA.py
import argparse, os, re, math, random, json
from contextlib import nullcontext
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import r2_score, mean_absolute_error

# Silence tokenizer fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------- Utils ----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def clean_text(s: str, max_chars=480):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:max_chars]

def to_float(x):
    try: return float(x)
    except: return np.nan

# ---------------------- Dataset ----------------------
REQUIRED_COLS = ["Series_Title", "meta_prompt", "overview", "Poster_Image_Path", "Gross"]

class MovieDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        self.df["Gross"] = self.df["Gross"].apply(to_float)
        self.df = self.df.dropna(subset=["meta_prompt", "overview", "Poster_Image_Path", "Gross"]).reset_index(drop=True)

        self.df["text1"] = self.df["meta_prompt"].apply(lambda s: clean_text(s, 480))
        self.df["text2"] = self.df["overview"].apply(lambda s: clean_text(s, 480))
        # We'll create y_log1p but NOT use it directly in the loss; we standardize later
        self.df["y_log1p"] = np.log1p(self.df["Gross"].astype(float))

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["Poster_Image_Path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(127, 127, 127))
        return {
            "image": img,
            "text1": row["text1"],
            "text2": row["text2"],
            "y_log1p": torch.tensor(row["y_log1p"], dtype=torch.float32),
        }

def collate_keep_pil(batch):
    return {
        "image":   [b["image"]   for b in batch],
        "text1":   [b["text1"]   for b in batch],
        "text2":   [b["text2"]   for b in batch],
        "y_log1p": torch.stack([b["y_log1p"] for b in batch], 0),
    }

# ---------------------- Model Head ----------------------
class MixerHead(nn.Module):
    """
    Fuse (v, t1, t2) with learnable softmax weights, MLP, then an affine output:
      y_std_hat = mlp(z)
      y_log1p_hat = y_std_hat * sigma + mu
    We set (mu, sigma) via .set_output_scaling(mu, sigma) after we compute them on the train set.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3) / 3)  # weights for [v, t1, t2]
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        # affine output params (learnable); initialize to identity
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.out_bias  = nn.Parameter(torch.tensor(0.0))

    def set_output_scaling(self, mu, sigma):
        # initialize to map mlp output ~0 to y ≈ mu; and scale ≈ sigma
        with torch.no_grad():
            self.out_scale.copy_(torch.tensor(float(sigma)))
            self.out_bias.copy_(torch.tensor(float(mu)))

    def forward(self, v, t1, t2, return_std=False):
        w = torch.softmax(self.w, dim=0)
        z = w[0]*v + w[1]*t1 + w[2]*t2
        z = nn.functional.normalize(z, dim=-1)
        y_std = self.mlp(z).squeeze(-1)  # standardized prediction
        y = y_std * self.out_scale + self.out_bias
        return (y, y_std) if return_std else y

# ---------------------- Encode / Train / Eval ----------------------
@torch.no_grad()
def encode_batch(clip, processor, images, texts1, texts2, device):
    # max token length for CLIP text (usually 77)
    try:
        max_len = clip.config.text_config.max_position_embeddings
    except Exception:
        max_len = 77

    # --- images ---
    img_inputs = processor.image_processor(images=images, return_tensors="pt")
    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
    v = clip.get_image_features(**img_inputs)
    v = nn.functional.normalize(v, dim=-1)

    # --- text1 ---
    t1_inputs = processor.tokenizer(
        texts1,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    t1_inputs = {k: v.to(device) for k, v in t1_inputs.items()}
    t1 = clip.get_text_features(**t1_inputs)
    t1 = nn.functional.normalize(t1, dim=-1)

    # --- text2 ---
    t2_inputs = processor.tokenizer(
        texts2,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    t2_inputs = {k: v.to(device) for k, v in t2_inputs.items()}
    t2 = clip.get_text_features(**t2_inputs)
    t2 = nn.functional.normalize(t2, dim=-1)

    return v, t1, t2

def run_epoch(dloader, clip, processor, head, optimizer, scheduler, device, mu, sigma, train=True):
    head.train() if train else head.eval()
    mse = nn.HuberLoss(delta=1.0)
    losses = []

    scaler = torch.amp.GradScaler('cuda') if device.startswith("cuda") else None
    amp_ctx = (lambda: torch.amp.autocast('cuda')) if device.startswith("cuda") else nullcontext

    for batch in dloader:
        images = batch["image"]
        y_log = batch["y_log1p"].to(device)
        # standardize targets
        y_std = (y_log - mu) / sigma

        with amp_ctx():
            with torch.no_grad():
                v, t1, t2 = encode_batch(clip, processor, images, batch["text1"], batch["text2"], device)
            # predict standardized, compute loss in standardized space
            y_pred, y_pred_std = head(v, t1, t2, return_std=True)
            loss = mse(y_pred_std, y_std)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        losses.append(loss.item())

    return float(np.mean(losses))

@torch.no_grad()
def evaluate_real_space(dloader, clip, processor, head, device, mape_floor=1e6):
    head.eval()
    preds, gts = [], []
    for batch in dloader:
        v, t1, t2 = encode_batch(clip, processor, batch["image"], batch["text1"], batch["text2"], device)
        yhat_log = head(v, t1, t2)
        preds.append(yhat_log.cpu().numpy())
        gts.append(batch["y_log1p"].cpu().numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    preds_real = np.expm1(preds); gts_real = np.expm1(gts)

    mae = mean_absolute_error(gts_real, preds_real)

    # filtered MAPE (ignore tiny targets where division blows up)
    mask = gts_real >= mape_floor
    if mask.sum() > 0:
        mape = np.mean(np.abs((gts_real[mask] - preds_real[mask]) / gts_real[mask]))
    else:
        mape = np.nan

    # sMAPE is robust when y is small
    smape = np.mean(2*np.abs(preds_real - gts_real) / (np.abs(preds_real) + np.abs(gts_real) + 1e-6))

    r2 = r2_score(gts_real, preds_real)
    return mae, mape, smape, r2

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True, help="Path to train.csv")
    ap.add_argument("--val",   type=str, required=True, help="Path to val.csv")
    ap.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                    help="HF model id (e.g., openai/clip-vit-base-patch32)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--gpu",    type=int, default=1, help="GPU index to use (default: 1)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    train_ds = MovieDataset(args.train)
    val_ds   = MovieDataset(args.val)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=collate_keep_pil)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=collate_keep_pil)

    # Compute μ, σ on TRAIN ONLY (in log-space)
    y_log_train = train_ds.df["y_log1p"].values.astype(np.float32)
    mu = torch.tensor(float(np.mean(y_log_train)), device=device)
    sigma = torch.tensor(float(np.std(y_log_train) + 1e-8), device=device)
    # Save scaling for later
    with open("scaling.json", "w") as f:
        json.dump({"mu": float(mu.item()), "sigma": float(sigma.item())}, f)

    # CLIP backbone (frozen)
    clip = CLIPModel.from_pretrained(args.model)
    # use_fast=True removes the "slow processor" warning
    processor = CLIPProcessor.from_pretrained(args.model)
    clip.to(device)
    for p in clip.parameters(): p.requires_grad = False

    # Keep BN in eval to avoid small-batch drift
    def bn_eval(m):
        if m.__class__.__name__ in ["BatchNorm2d","FrozenBatchNorm2d"]:
            m.eval()
    if hasattr(clip, "vision_model"):
        clip.vision_model.apply(bn_eval)

    head = MixerHead(clip.config.projection_dim).to(device)
    head.set_output_scaling(mu.item(), sigma.item())

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.05,
                                  betas=(0.9,0.98), eps=1e-6)
    total_steps = max(1, len(train_dl) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(train_dl, clip, processor, head, optimizer, scheduler, device, mu, sigma, train=True)
        va_loss = run_epoch(val_dl,   clip, processor, head, optimizer, scheduler, device, mu, sigma, train=False)
        mae, mape, smape, r2 = evaluate_real_space(val_dl, clip, processor, head, device)
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"val_MAE={mae:,.2f}  val_fMAPE(>=$1M)={100*mape if mape==mape else float('nan'):.2f}%  "
              f"val_sMAPE={100*smape:.2f}%  val_R2={r2:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"head": head.state_dict()}, "head_stageA.pt")
            print("  ↳ saved head_stageA.pt")

if __name__ == "__main__":
    main()