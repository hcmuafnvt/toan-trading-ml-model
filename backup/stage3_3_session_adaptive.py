import os, numpy as np, pandas as pd, lightgbm as lgb, vectorbt as vbt
from ta.volatility import AverageTrueRange
from datetime import timezone

# ---------- CONFIG ----------
PAIR = "GBP_USD"
DATA_FILE = f"data/{PAIR}_M5_2024.parquet"
FEATURE_FILE = "logs/stage2_features.csv"
MODEL_FILES = {
    "T1_10x40": "logs/T1_10x40_lightgbm.txt",
    "T2_15x60": "logs/T2_15x60_lightgbm.txt",
    "T3_20x80": "logs/T3_20x80_lightgbm.txt",
}
LOW_Q, HIGH_Q = 0.33, 0.66
CONF_TH = 0.3
PIP_SIZE, PIP_USD = 0.0001, 10
OUT_TXT = "logs/stage3_3_summary.txt"
os.makedirs("logs", exist_ok=True)

# ---------- HELPERS ----------
def clean(df):
    df = df.copy()
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]+","_",regex=True).str.strip("_")
    return df

def load_price(path):
    df = pd.read_parquet(path)
    df["close"] = df["mid_c"]; df["open"]=df["mid_o"]; df["high"]=df["mid_h"]; df["low"]=df["mid_l"]
    df = df[["open","high","low","close","volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def atr_from_ta(price, win=14):
    tmp = price[["high","low","close"]]
    a = AverageTrueRange(tmp["high"],tmp["low"],tmp["close"],window=win).average_true_range()
    a.index = price.index; return a.ffill().bfill()

def get_session(t):
    h = t.hour
    if 22<=h or h<6: return "Asia"
    elif 6<=h<14: return "London"
    else: return "NewYork"

def get_conf_margin(probs):
    s = np.sort(probs,axis=1)
    return s[:,-1]-s[:,-2]

def scale(x,lo=0.5,hi=2.0):
    a,b = np.nanmin(x),np.nanmax(x)
    return lo+(x-a)*(hi-lo)/(b-a+1e-12)

# ---------- MAIN ----------
print(f"⏳ Load price {DATA_FILE}")
price = load_price(DATA_FILE)
atr = atr_from_ta(price)
qL,qH = atr.quantile(LOW_Q), atr.quantile(HIGH_Q)
regime = np.where(atr<=qL,"Low",np.where(atr>=qH,"High","Normal"))
session = price.index.map(get_session)
price["regime"],price["session"]=regime,session
print("✅ Regime/session done:", pd.crosstab(price["session"],price["regime"]).to_string())

feat = clean(pd.read_csv(FEATURE_FILE)).replace([np.inf,-np.inf],0)
target_cols=[c for c in feat.columns if c.startswith("target_")]
X=feat.drop(columns=target_cols,errors="ignore")
n=len(X)
sample_idx=np.arange(200,len(price),5)[:n]
sample_times=price.index[sample_idx]
probs_all={}
for k,v in MODEL_FILES.items():
    b=lgb.Booster(model_file=v)
    pr=b.predict(X.values,num_iteration=b.best_iteration)
    probs_all[k]=pr
probs=np.stack(list(probs_all.values()),axis=0).sum(axis=0)
pred=np.argmax(probs,axis=1)
sig=pd.Series(pred,index=sample_times).reindex(price.index,method="ffill").fillna(1).astype(int)
conf=pd.Series(get_conf_margin(probs),index=sample_times).reindex(price.index,method="ffill").fillna(0)
size=pd.Series(scale(conf.values),index=price.index)
sig[conf<CONF_TH]=1   # filter weak confidence

# adaptive TP/SL by regime
tp=pd.Series(20*PIP_SIZE,index=price.index)
sl=pd.Series(20*PIP_SIZE,index=price.index)
atr14=atr_from_ta(price)
tp[price["regime"]=="High"]=(atr14*2.0).clip(lower=1e-12)
sl[price["regime"]=="High"]=(atr14*2.0).clip(lower=1e-12)

# run per session
def backtest(mask):
    s=sig.copy(); s[~mask]=1
    pf=vbt.Portfolio.from_signals(price["close"],
        entries=s.eq(2), exits=~s.eq(2),
        short_entries=s.eq(0), short_exits=~s.eq(0),
        tp_stop=tp, sl_stop=sl, size=size, freq="5min")
    st=pf.stats()
    exp=np.mean((pf.trades.records["exit_price"]-pf.trades.records["entry_price"])*np.where(pf.trades.records["direction"]==0,1,-1)/PIP_SIZE)
    return st,exp

rows=[]
for ss in ["Asia","London","NewYork"]:
    for rg in ["Low","Normal","High"]:
        mask=(price["session"]==ss)&(price["regime"]==rg)
        if mask.sum()<1000: continue
        st,exp=backtest(mask)
        pf=st["Profit Factor"]; win=st["Win Rate [%]"]; ret=st["Total Return [%]"]
        rows.append([ss,rg,pf,win,exp,mask.sum()])
        print(f"{ss:<8}{rg:<8} PF={pf:.2f} Win={win:.1f}% Exp={exp:.2f}p n={mask.sum()}")

res=pd.DataFrame(rows,columns=["Session","Regime","PF","Win%","Exp(p)","Samples"])
res.to_csv("logs/stage3_3_grid.csv",index=False)

st_all, exp_all = backtest(price["close"] > 0)
print("\n========== FINAL PORTFOLIO ==========")
print(st_all.to_string())
print(f"Expectancy={exp_all:.2f}p ($ {exp_all*PIP_USD:.2f})")
with open(OUT_TXT,"w") as f: f.write(st_all.to_string())