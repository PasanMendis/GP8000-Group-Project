# 3_eval.py
import torch, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from 1_train_clip_regressor import MovieDataset, MixerHead, encode_batch

MODEL_ID = "openai/clip-resnet-50"
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = MovieDataset("test.csv")
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
clip = CLIPModel.from_pretrained(MODEL_ID).to(device)
proc = CLIPProcessor.from_pretrained(MODEL_ID)
head = MixerHead(emb=clip.config.projection_dim).to(device)
head.load_state_dict(torch.load("head_stageA.pt")["head"])  # or model_stageB.pt if you trained Stage-B
clip.eval(); head.eval()

preds, gts = [], []
with torch.no_grad():
    for batch in dl:
        v,t1,t2 = encode_batch(batch["image"], batch["text1"], batch["text2"])
        yhat_log = head(v,t1,t2)
        preds.append(yhat_log.cpu().numpy())
        gts.append(batch["y_log1p"].cpu().numpy())
preds = np.concatenate(preds); gts = np.concatenate(gts)

# back to real space
preds_real = np.expm1(preds)
gts_real   = np.expm1(gts)

mae = mean_absolute_error(gts_real, preds_real)
mape = np.mean(np.abs((gts_real - preds_real) / np.maximum(1e-6, gts_real)))
r2 = r2_score(gts_real, preds_real)
print(f"Test  MAE: {mae:,.2f}   MAPE: {100*mape:.2f}%   R²: {r2:.3f}")