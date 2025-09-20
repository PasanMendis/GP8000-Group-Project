# quick_check.py
import pandas as pd, numpy as np

df = pd.read_csv("val.csv")
y = df["Gross"].astype(float)

print("VAL target stats:")
print("count:", y.size)
print("zeros:", int((y==0).sum()))
print("min:", y.min(), "median:", y.median(), "mean:", y.mean(), "p90:", y.quantile(0.9))

# naive baseline (predict train mean in log space)
tr = pd.read_csv("train.csv")
mu = np.log1p(tr["Gross"].astype(float)).mean()
pred = np.expm1(np.full_like(y, mu))

mae = np.mean(np.abs(y - pred))
mape = np.mean(np.abs((y - pred) / np.maximum(1e-6, y)))  # explodes if many y≈0
def smape(a, f): return np.mean(2*np.abs(f-a)/(np.abs(a)+np.abs(f)+1e-6))
print(f"Baseline MAE: {mae:,.0f}")
print(f"Baseline MAPE: {100*mape:.2f}%")
print(f"Baseline sMAPE: {100*smape(y.values, pred):.2f}%")