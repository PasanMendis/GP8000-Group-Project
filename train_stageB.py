# train_stageB_3text.py
import argparse, os, re, math, random, warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# ------------- utils -------------
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
    """
    Accepts numeric or string. Returns a short, clean budget string for CLIP text.
      - 120000000 -> "Budget: $120,000,000"
      - "$120M" / "0.15B" -> normalized similarly
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "Budget: unknown"
    s = str(x).strip()
    # Try numeric first
    val = None
    try:
        # raw like 120000000 or "120,000,000"
        val = float(s.replace(",", "").replace("$",""))
    except Exception:
        # suffix M/B
        m = re.match(r"^\$?\s*([0-9]*\.?[0-9]+)\s*([MmBb])$", s)
        if m:
            val = float(m.group(1))
            if m.group(2).upper() == "M": val *= 1e6
            else: val *= 1e9
    if val is None:
        return f"Budget: {clean_text(s, 48)}"
    # pretty format with commas
    return f"Budget: ${int(val):,}"

# ------------- dataset -------------
REQUIRED = ["Series_Title", "meta_prompt", "overview", "Poster_Image_Path", "Gross", "budget"]

class MovieDataset(Dataset):
    """
    Uses 3 texts: meta_prompt, overview, budget_text; and the poster image.
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        miss = [c for c in REQUIRED if c not in df.columns]
        if miss:
            raise ValueError(f"Missing columns in {csv_path}: {miss}")

        # coerce essentials
        df["Gross"] = df["Gross"].apply(to_float)
        df = df.dropna(subset=["Poster_Image_Path", "Gross"]).reset_index(drop=True)

        # text columns
        df["meta_prompt"] = df["meta_prompt"].apply(lambda s: clean_text(s, 480))
        df["overview"]    = df["overview"].apply(lambda s: clean_text(s, 480))
        df["budget_text"] = df["budget"].apply(format_budget_as_text)

        # target (log1p)
        df["y_log1p"] = np.log1p(df["Gross"].astype(float))

        self.df = df

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        try:
            img = Image.open(r["Poster_Image_Path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224,224), (127,127,127))
        return {
            "image": img,
            "meta_prompt": r["meta_prompt"],
            "overview": r["overview"],
            "budget_text": r["budget_text"],
            "y_log1p": torch.tensor(r["y_log1p"], dtype=torch.float32),
        }

def collate_keep_pil(batch):
    return {
        "image":        [b["image"] for b in batch],
        "meta_prompt":  [b["meta_prompt"] for b in batch],
        "overview":     [b["overview"] for b in batch],
        "budget_text":  [b["budget_text"] for b in batch],
        "y_log1p":      torch.stack([b["y_log1p"] for b in batch], 0),
    }

# ------------- head (fusion over 1 image + 3 texts) -------------
class FusionHead(nn.Module):
    """
    Weighted fusion over [image, t_meta, t_overview, t_budget] -> MLP -> standardized log target.
    """
    def __init__(self, emb_dim, drop1=0.35, drop2=0.35):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.25, 0.35, 0.30, 0.10]))  # init prior: meta > overview > image ~ budget
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 1024), nn.GELU(), nn.Dropout(drop1),
            nn.Linear(1024, 256), nn.GELU(), nn.Dropout(drop2),
            nn.Linear(256, 1),
        )
        self.out_scale = nn.Parameter(torch.tensor(1.0))  # sigma
        self.out_bias  = nn.Parameter(torch.tensor(0.0))  # mu

    def set_output_scaling(self, mu, sigma):
        with torch.no_grad():
            self.out_scale.copy_(torch.tensor(float(sigma)))
            self.out_bias.copy_(torch.tensor(float(mu)))

    def forward(self, v, t_meta, t_over, t_budget, return_std=False):
        w = torch.softmax(self.w, dim=0)
        z = w[0]*v + w[1]*t_meta + w[2]*t_over + w[3]*t_budget
        z = nn.functional.normalize(z, dim=-1)
        y_std = self.mlp(z).squeeze(-1)
        y_log = y_std * self.out_scale + self.out_bias
        return (y_log, y_std) if return_std else y_log

# ------------- encode -------------
@torch.no_grad()
def encode_batch(clip, processor, images, meta, over, budg, device):
    try:
        max_len = clip.config.text_config.max_position_embeddings
    except Exception:
        max_len = 77

    # images
    img_inputs = processor.image_processor(images=images, return_tensors="pt")
    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
    v = clip.get_image_features(**img_inputs)
    v = nn.functional.normalize(v, dim=-1)

    # text encodes
    def enc_text(texts):
        toks = processor.tokenizer(texts, return_tensors="pt",
                                   padding=True, truncation=True, max_length=max_len)
        toks = {k: v.to(device) for k, v in toks.items()}
        t = clip.get_text_features(**toks)
        return nn.functional.normalize(t, dim=-1)

    t_meta = enc_text(meta)
    t_over = enc_text(over)
    t_budg = enc_text(budg)
    return v, t_meta, t_over, t_budg

# ------------- eval metrics (real $) -------------
@torch.no_grad()
def evaluate(loader, clip, processor, head, device):
    preds, gts = [], []
    for b in loader:
        v, t1, t2, t3 = encode_batch(clip, processor, b["image"], b["meta_prompt"], b["overview"], b["budget_text"], device)
        y_log = head(v, t1, t2, t3)
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

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_enc",  type=float, default=5e-6)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--gpu",    type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # data
    train_ds = MovieDataset(args.train)
    val_ds   = MovieDataset(args.val)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_keep_pil)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_keep_pil)

    # target scaling (log space)
    y_log_train = train_ds.df["y_log1p"].values.astype(np.float32)
    mu_y = float(y_log_train.mean())
    sd_y = float(y_log_train.std() + 1e-8)

    # backbone
    clip = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    # head
    head = FusionHead(emb_dim=clip.config.projection_dim, drop1=0.35, drop2=0.35).to(device)
    head.set_output_scaling(mu_y, sd_y)

    # partial unfreeze
    for p in clip.parameters(): p.requires_grad = False
    try:
        for blk in clip.vision_model.encoder.layers[-2:]:
            for p in blk.parameters(): p.requires_grad = True
    except Exception: pass
    try:
        for blk in clip.text_model.encoder.layers[-2:]:
            for p in blk.parameters(): p.requires_grad = True
    except Exception: pass
    for p in clip.visual_projection.parameters(): p.requires_grad = True
    for p in clip.text_projection.parameters():   p.requires_grad = True

    def bn_eval(m):
        if m.__class__.__name__ in ["BatchNorm2d","FrozenBatchNorm2d"]:
            m.eval()
    if hasattr(clip, "vision_model"): clip.vision_model.apply(bn_eval)

    # optimizer / sched / loss
    enc_params = [p for p in clip.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": args.lr_head, "weight_decay": 0.05},
         {"params": enc_params,        "lr": args.lr_enc,  "weight_decay": 0.01}],
        betas=(0.9,0.98), eps=1e-6
    )
    total_steps = max(1, len(train_dl)*args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05*total_steps), total_steps)
    loss_fn = nn.HuberLoss(delta=1.0)

    scaler = torch.amp.GradScaler('cuda') if device.startswith("cuda") else None
    amp_ctx = (lambda: torch.amp.autocast('cuda')) if device.startswith("cuda") else nullcontext

    best_mae = float("inf")
    for epoch in range(1, args.epochs+1):
        head.train(); clip.train()
        tr_losses = []
        for b in train_dl:
            y_log = b["y_log1p"].to(device)
            # standardize target in log space
            y_std = (y_log - mu_y) / sd_y
            with amp_ctx():
                v, t_meta, t_over, t_budg = encode_batch(clip, processor, b["image"], b["meta_prompt"], b["overview"], b["budget_text"], device)
                y_log_pred, y_std_pred = head(v, t_meta, t_over, t_budg, return_std=True)
                loss = loss_fn(y_std_pred, y_std)

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

        # eval
        head.eval(); clip.eval()
        mae, medae, rmse, fmape, smape, r2 = evaluate(val_dl, clip, processor, head, device)
        print(f"[Stage-B(3text) {epoch}/{args.epochs}] "
              f"train_loss={np.mean(tr_losses):.4f}  "
              f"val_MAE={mae:,.2f}  val_MedAE={medae:,.2f}  val_RMSE={rmse:,.2f}  "
              f"val_fMAPE(>=$1M)={(100*fmape if fmape==fmape else float('nan')):.2f}%  "
              f"val_sMAPE={100*smape:.2f}%  val_R2={r2:.3f}")

        if mae < best_mae:
            best_mae = mae
            torch.save({
                "head": head.state_dict(), "clip": clip.state_dict(),
                "mu_y": mu_y, "sd_y": sd_y
            }, "model_stageB_3text.pt")
            print("  ↳ saved model_stageB_3text.pt (best MAE)")

if __name__ == "__main__":
    main()