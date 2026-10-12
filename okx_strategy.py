"""
OKX Event Contract 策略 —— 自包含复现文件
==========================================
用 OKX 原生数据训练 5min t+2 涨跌预测模型(用于 event contract)。

数据输入 (4个 OKX CSV, 用附带的 okx_index_collect.py / okx_spot_collect.py 拉取):
  现货(特征): okx_spot_ETH-USDT_5m.csv, okx_spot_BTC-USDT_5m.csv
      列: ts,dt,open,high,low,close,vol,volCcy,volCcyQuote
  指数(标签): okx_index_ETH-USDT_5m.csv, okx_index_BTC-USDT_5m.csv
      列: ts,dt,open,high,low,close

用法:
  python okx_strategy.py train      # 训练+保存模型
  python okx_strategy.py backtest   # 回测(打印各档命中率/EV/净值统计)
  python okx_strategy.py signal      # 对最新一根出信号(押UP/DOWN/跳过)

依赖: pip install numpy pandas lightgbm scikit-learn joblib
"""
import sys, os
import numpy as np, pandas as pd

# ============ 配置 ============
DATA_DIR = "."                      # CSV 所在目录
SPOT = {'ETH': "../data/okx_spot_ETH-USDT_5m.csv", 'BTC': "../data/okx_spot_BTC-USDT_5m.csv"}
INDEX = {'ETH': "../data/okx_index_ETH-USDT_5m.csv", 'BTC': "../data/okx_index_BTC-USDT_5m.csv"}
LAG = 2                             # 预测 t+2 (下下个5min窗口)
SPLIT = pd.Timestamp('2025-06-01')  # 训练/测试切分
CONF_THRESH = 0.04                  # 下注置信阈值 (ETH主力档)
ENTRY = 0.51                        # 假设入场价
STAKE = 32                          # 每注 USDT
MODEL_FILE = "okx_model_pool.pkl"
PARAMS = dict(num_leaves=15, min_child_samples=1000, learning_rate=0.01,
              n_estimators=1000, reg_lambda=5.0, subsample=0.7, colsample_bytree=0.7)
N_SEEDS = 5

# ============ 数据加载 ============
def load_okx(coin):
    sp = pd.read_csv(os.path.join(DATA_DIR, SPOT[coin]))
    ix = pd.read_csv(os.path.join(DATA_DIR, INDEX[coin]))
    sp['dt'] = pd.to_datetime(sp['dt']); ix['dt'] = pd.to_datetime(ix['dt'])
    sp = sp.set_index('dt').sort_index().rename(columns={
        'open':'Open','high':'High','low':'Low','close':'Close',
        'vol':'Volume','volCcyQuote':'Quote asset volume'})
    ix = ix.set_index('dt').sort_index()[['open','close']].rename(
        columns={'open':'idx_open','close':'idx_close'})
    df = sp.join(ix, how='inner')
    return df.dropna(subset=['Close','idx_close'])

# ============ 特征 (47个, 无订单流) ============
def feats(d):
    c,h,l,v = d['Close'],d['High'],d['Low'],d['Volume']
    r = c.pct_change(); lr = np.log(c).diff()
    X = pd.DataFrame(index=d.index)
    for k in [1,2,3,4,6,9,12]: X[f'ret_lag{k}'] = r.shift(k-1) if k==1 else r.rolling(k).sum()
    for k in [1,2,3,6,12]: X[f'r{k}'] = r.rolling(k).sum()
    for k in [12,48,144]: X[f'vol{k}'] = lr.rolling(k).std()
    X['vol_ratio'] = X['vol12']/X['vol144']
    X['range'] = (h-l)/c; X['range_ma'] = X['range'].rolling(12).mean()
    X['close_pos'] = (c-l)/(h-l+1e-12)
    for k in [12,48,144]:
        ma = c.rolling(k).mean(); X[f'dev{k}'] = (c-ma)/c.rolling(k).std()
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    X['rsi'] = 100-100/(1+up/dn.replace(0,np.nan))
    X['vol_z'] = (v-v.rolling(48).mean())/v.rolling(48).std()
    X['vol_chg'] = v.pct_change().clip(-5,5)
    tod = d.index.hour*60+d.index.minute
    X['tod_sin'] = np.sin(2*np.pi*tod/1440); X['tod_cos'] = np.cos(2*np.pi*tod/1440)
    X['dow'] = d.index.dayofweek
    sgn = np.sign(r)
    for k in [3,6,12]: X[f'streak{k}'] = sgn.rolling(k).sum()
    X['last_sign'] = sgn
    for k in [12,48,144]:
        hh = h.rolling(k).max(); ll = l.rolling(k).min(); X[f'donch_pos{k}'] = (c-ll)/(hh-ll+1e-12)
    X['mtf_align'] = np.sign(c.pct_change(12))+np.sign(c.pct_change(48))
    pk = np.log(h/l)**2; X['parkinson12'] = np.sqrt(pk.rolling(12).mean()/(4*np.log(2)))
    X['updown_vol'] = lr.clip(lower=0).rolling(48).std()/(lr.clip(upper=0).abs().rolling(48).std()+1e-9)
    X['skew48'] = lr.rolling(48).skew(); X['kurt48'] = lr.rolling(48).kurt()
    X['rvol'] = v/v.rolling(288).mean()
    X['accel'] = r-r.shift(1); X['vol_accel'] = X['vol12'].diff()
    vwap = (d['Quote asset volume'].rolling(48).sum()/v.rolling(48).sum())
    X['vwap_dev'] = (c-vwap)/c.rolling(48).std()
    how = d.index.dayofweek*24+d.index.hour
    X['how_sin'] = np.sin(2*np.pi*how/168); X['how_cos'] = np.cos(2*np.pi*how/168)
    X['is_hour_open'] = (d.index.minute==0).astype(int)
    return X

def target(d, lag=LAG):
    return (d['idx_close'].shift(-lag) >= d['idx_open'].shift(-lag)).astype(int)

def build_dataset(coin):
    d = load_okx(coin); X = feats(d); X['y'] = target(d)
    data = X.replace([np.inf,-np.inf], np.nan).dropna()
    return data

# ============ 训练 ============
def train():
    import lightgbm as lgb, joblib
    Ps = {c: build_dataset(c) for c in ['ETH','BTC']}
    feat_cols = [c for c in Ps['ETH'].columns if c != 'y']
    pool = pd.concat([Ps[c][Ps[c].index < SPLIT] for c in Ps])
    print(f"池化训练集: {len(pool)} 行, {len(feat_cols)} 特征, 切分 {SPLIT.date()}")
    models = []
    for s in range(N_SEEDS):
        m = lgb.LGBMClassifier(random_state=s, verbose=-1, **PARAMS)
        m.fit(pool[feat_cols], pool['y'])
        models.append(m)
        print(f"  种子 {s} 训练完成")
    joblib.dump({'models': models, 'feat_cols': feat_cols, 'params': PARAMS}, MODEL_FILE)
    print(f"已保存 -> {MODEL_FILE}")

def predict_proba(data, feat_cols, models):
    ps = [m.predict_proba(data[feat_cols])[:,1] for m in models]
    return pd.Series(np.mean(ps, axis=0), index=data.index)

# ============ 回测 ============
def backtest(coin='ETH'):
    import joblib
    bundle = joblib.load(MODEL_FILE)
    data = build_dataset(coin)
    te = data[data.index >= SPLIT]
    p = predict_proba(te, bundle['feat_cols'], bundle['models'])
    y = te['y']; win = ((p>0.5).astype(int)==y).astype(int); conf = (p-0.5).abs()
    days = (te.index[-1]-te.index[0]).days; yr = 365/days
    shares = STAKE/ENTRY; taker_fee = shares*(4*1.125*ENTRY*(1-ENTRY))/100
    print(f"\n=== 回测 {coin} 5min t+2 | 测试 {te.index.min().date()}~{te.index.max().date()} ===")
    from sklearn.metrics import roc_auc_score
    print(f"总命中={win.mean():.4f}  AUC={roc_auc_score(y,p):.4f}")
    print(f"入场{ENTRY} taker含费 每注{STAKE}u(费{taker_fee:.3f}u)")
    print("阈值 | 命中率 | 含费EV/注 | 年盈利@%du | 亏损月 | 月夏普" % STAKE)
    for tau in [0.0, 0.04, 0.06]:
        m = (conf>tau).values
        if m.sum()<20: continue
        w = win[m]; h = w.mean()
        pnl = np.where(w==1, shares*(1-ENTRY)-taker_fee, -shares*ENTRY-taker_fee)
        ev = pnl.mean()/STAKE*100; annu = pnl.sum()*yr
        mp = pd.Series(pnl, index=te.index[m]).groupby(te.index[m].to_period('M')).sum()
        sm = mp.mean()/mp.std()*np.sqrt(12) if mp.std()>0 else 0
        print(f">{tau:.2f} | {h:.4f} | {ev:+.2f}% | {annu:>8,.0f} | {int((mp<0).sum())}/{len(mp)} | {sm:.2f}")

# ============ 实盘信号(对最新一根) ============
def signal(coin='ETH'):
    import joblib
    bundle = joblib.load(MODEL_FILE)
    d = load_okx(coin)
    X = feats(d).replace([np.inf,-np.inf], np.nan)
    latest = X.dropna().iloc[[-1]]      # 最新一根(已收盘)的特征
    p = predict_proba(latest, bundle['feat_cols'], bundle['models']).iloc[0]
    conf = abs(p-0.5)
    direction = "UP" if p>0.5 else "DOWN"
    t = latest.index[0]
    print(f"\n=== 信号 {coin} @ {t} (最新已收盘K线) ===")
    print(f"预测目标: t+{LAG} 窗口 (约 {t + pd.Timedelta(minutes=5*LAG)} ~ {t + pd.Timedelta(minutes=5*(LAG+1))})")
    print(f"模型概率(涨)={p:.4f}  置信度={conf:.4f}")
    if conf > CONF_THRESH:
        print(f">>> 下注: {direction}  (置信 {conf:.4f} > {CONF_THRESH})")
    else:
        print(f">>> 跳过 (置信 {conf:.4f} <= {CONF_THRESH})")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv)>1 else "backtest"
    coin = sys.argv[2] if len(sys.argv)>2 else "ETH"
    if cmd == "train": train()
    elif cmd == "backtest": backtest(coin)
    elif cmd == "signal": signal(coin)
    else: print("用法: python okx_strategy.py [train|backtest|signal] [ETH|BTC]")