"""
بيانات اصطناعية لاختبار المنطق بدون إنترنت. الرموز DEMO* وهمية.
Synthetic scenarios: each one is built to look like a pattern from the guide.
"""
from datetime import datetime, time, timedelta
import numpy as np
import pandas as pd

from .data_sources import ET


def _dates(n, end):
    days = pd.bdate_range(end=end, periods=n)
    return [pd.Timestamp(datetime.combine(d.date(), time(0, 0)), tz=ET) for d in days]


def make_daily(path, end, vol_base=400_000, seed=0, overrides=None):
    rng = np.random.default_rng(seed)
    n = len(path)
    rows = []
    prev = path[0]
    for i, c in enumerate(path):
        c = c * (1 + rng.normal(0, 0.012))
        o = prev * (1 + rng.normal(0, 0.004))
        hi = max(o, c) * (1 + abs(rng.normal(0, 0.012)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.012)))
        v = vol_base * (1 + abs(rng.normal(0, 0.3)))
        rows.append([o, hi, lo, c, v])
        prev = c
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=_dates(n, end))
    for i, vals in (overrides or {}).items():
        df.iloc[i] = vals
    return df


def add_runner(df, bars_ago=170, mult=1.7):
    """انفجار قديم (Former Runner): شمعة +70% بفوليوم كبير قبل ~8 أشهر."""
    i = max(20, len(df) - bars_ago)
    pc = float(df["close"].iloc[i - 1])
    df.iloc[i] = [pc, pc * mult, pc * 0.98, pc * 1.35, float(df["volume"].iloc[i - 1]) * 12]
    return df


def seg(a, b, n):
    return list(np.linspace(a, b, n + 1)[1:])


def scenario_bounce(end):
    # هبوط طويل ← دعم 1.20 لمس مرتين ← تماسك هادئ ← بيع حاد للدعم (RSI ~24) ← سحب سيولة بمطرقة واضحة
    flat, nd = 15, 6
    path = [4.0] + seg(4.0, 1.60, 100)
    path += seg(1.60, 1.21, 12) + seg(1.21, 1.42, 8) + seg(1.42, 1.22, 9) + seg(1.22, 1.50, 10)
    path += [1.50] * (flat + nd + 4)
    df = make_daily(path, end, seed=1)
    v = 400_000
    start = -(flat + nd + 4)
    for k in range(flat):
        x = 1.50 * (1 + (0.004 if k % 2 else -0.004))
        df.iloc[start + k] = [1.50, 1.505, 1.495, x, v * 0.7]
    prev = df.iloc[start + flat - 1]["close"]
    for k, cl in enumerate(np.linspace(1.43, 1.22, nd)):
        df.iloc[start + flat + k] = [prev, prev * 1.003, cl * 0.995, cl, v * 1.5]
        prev = cl
    df.iloc[-4] = [1.22, 1.226, 1.212, 1.218, v * 0.8]
    df.iloc[-3] = [1.218, 1.224, 1.208, 1.216, v * 0.6]
    df.iloc[-2] = [1.216, 1.222, 1.205, 1.214, v * 0.5]
    df.iloc[-1] = [1.212, 1.275, 1.17, 1.268, v * 1.8]      # مطرقة: كسر بسيط للدعم ثم إغلاق قوي بفوليوم أعلى
    return add_runner(df)


def scenario_sweep(end):
    path = [3.0] + seg(3.0, 1.10, 110) + seg(1.10, 0.90, 10) + seg(0.90, 1.05, 8) + seg(1.05, 0.905, 8)
    path += seg(0.905, 1.10, 10) + seg(1.10, 1.03, 6) + seg(1.03, 0.92, 8)
    df = make_daily(path + [0.915, 0.915], end, seed=2)
    v = 500_000
    df.iloc[-2] = [0.915, 0.925, 0.858, 0.912, v * 1.8]      # كسر بسيط للدعم 0.90 ثم إغلاق فوقه
    df.iloc[-1] = [0.912, 0.965, 0.905, 0.958, v * 1.5]      # استرداد وشمعة خضراء
    return df


def scenario_w(end):
    path = [2.5] + seg(2.5, 1.20, 100) + seg(1.20, 1.00, 12)          # L1 = 1.00
    path += seg(1.00, 1.25, 8) + seg(1.25, 1.02, 9)                    # خط العنق 1.25 و L2 = 1.02
    path += seg(1.02, 1.22, 7) + [1.27, 1.29, 1.28]                    # كسر وثبات
    df = make_daily(path, end, seed=3)
    df.iloc[-3] = [1.21, 1.31, 1.20, 1.27, 900_000]
    df.iloc[-2] = [1.27, 1.31, 1.255, 1.29, 600_000]
    df.iloc[-1] = [1.29, 1.30, 1.262, 1.285, 500_000]
    return add_runner(df, bars_ago=100)


def scenario_base_flag(end):
    """🚩 ص79 مثال 4: هبوط → قاعدة (RSI ~25) → اختراق مقاومة قريبة بفوليوم → إعادة اختبار ناجحة."""
    path = [2.6] + seg(2.6, 1.05, 100) + [1.02, 1.00, 1.01, 1.03, 1.05, 1.07, 1.10, 1.08, 1.05, 1.04, 1.06, 1.05,
                                         1.07, 1.06, 1.08]
    path += [1.18, 1.22, 1.20, 1.15, 1.17]
    df = make_daily(path, end, seed=11)
    n = len(df)
    df.iloc[n - 14] = [1.08, 1.12, 1.07, 1.10, 450_000]       # المقاومة القريبة 1.12 (قمة القاعدة)
    for k in range(n - 13, n - 5):                             # القاعدة تحت المقاومة
        o, h, l, c, v = df.iloc[k]
        df.iloc[k] = [min(o, 1.08), min(h, 1.09), l, min(c, 1.08), v]
    df.iloc[-5] = [1.08, 1.21, 1.07, 1.18, 1_400_000]          # كسر بفوليوم
    df.iloc[-4] = [1.18, 1.25, 1.17, 1.22, 700_000]
    df.iloc[-3] = [1.22, 1.23, 1.18, 1.20, 500_000]
    df.iloc[-2] = [1.19, 1.20, 1.13, 1.15, 420_000]            # إعادة اختبار 1.12–1.15
    df.iloc[-1] = [1.15, 1.18, 1.14, 1.17, 520_000]            # شمعة ثبات خضراء فوق المقاومة
    return add_runner(df, bars_ago=100)


def scenario_pump(end):
    path = [2.0] + seg(2.0, 0.80, 110) + seg(0.80, 0.70, 20) + [0.71, 0.70, 0.72]
    df = make_daily(path, end, seed=4, vol_base=300_000)
    df.iloc[-2] = [0.72, 1.60, 0.71, 1.05, 6_000_000]       # شمعة قروبات وهم
    df.iloc[-1] = [1.02, 1.08, 0.86, 0.88, 2_500_000]
    return df


def scenario_failed(end):
    path = [3.0] + seg(3.0, 1.30, 100) + seg(1.30, 1.20, 8) + seg(1.20, 1.40, 8) + seg(1.40, 1.205, 8)
    path += seg(1.205, 1.35, 6) + seg(1.35, 1.12, 6) + [1.10, 1.09]
    return make_daily(path, end, seed=5)


def scenario_extended(end):
    path = [3.0] + seg(3.0, 1.10, 110) + seg(1.10, 0.95, 10) + seg(0.95, 1.10, 6) + seg(1.10, 0.96, 6)
    path += seg(0.96, 1.30, 5)
    return make_daily(path, end, seed=6)


def momentum_intraday(today, prev_close=2.0):
    """فجوة +30% ثم موجة أولى إلى 3.20 ثم Pullback إلى الفواب وثبات."""
    idx, rows = [], []
    t = datetime.combine(today, time(9, 30), tzinfo=ET)
    px = 2.60
    plan = seg(2.60, 3.20, 18) + seg(3.20, 2.95, 14) + [2.96, 2.97, 2.975]
    for i, c in enumerate(plan):
        o = px
        hi, lo = max(o, c) * 1.004, min(o, c) * 0.996
        vol = 400_000 if i < 18 else 150_000
        rows.append([o, hi, lo, c, vol])
        idx.append(pd.Timestamp(t + timedelta(minutes=5 * i)))
        px = c
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def demo_data(now_et):
    today = now_et.date()
    end = pd.Timestamp(today)
    daily = {
        "DEMOA": scenario_bounce(end),
        "DEMOB": scenario_sweep(end),
        "DEMOC": scenario_w(end),
        "DEMOD": scenario_pump(end),
        "DEMOE": scenario_failed(end),
        "DEMOF": scenario_extended(end),
    }
    # سهم زخم: تاريخ يومي + شمعة اليوم من البيانات اللحظية
    mdaily = make_daily([1.6] + seg(1.6, 2.0, 79), end - pd.Timedelta(days=1), seed=7, vol_base=300_000)
    intr = momentum_intraday(today)
    bar = pd.DataFrame({"open": [intr["open"].iloc[0]], "high": [intr["high"].max()], "low": [intr["low"].min()],
                        "close": [intr["close"].iloc[-1]], "volume": [intr["volume"].sum()]},
                       index=[pd.Timestamp(datetime.combine(today, time(0, 0)), tz=ET)])
    daily["DEMOM"] = pd.concat([mdaily[[d.date() < today for d in mdaily.index]], bar])
    intraday5 = {"DEMOM": intr}
    # 15 دقيقة لتأكيد الثبات (اصطناعي): آخر شمعتين فوق المستوى
    intraday15 = {}
    for k in ("DEMOA", "DEMOB", "DEMOC"):
        last = float(daily[k]["close"].iloc[-1])
        t0 = datetime.combine(today, time(9, 30), tzinfo=ET)
        rows = [[last, last * 1.01, last * 0.99, last * (1 + 0.002 * j), 50_000] for j in range(4)]
        intraday15[k] = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                                     index=[pd.Timestamp(t0 + timedelta(minutes=15 * j)) for j in range(4)])
    universe = pd.DataFrame([{"symbol": s, "name": f"Demo {s}", "price": float(d["close"].iloc[-1]), "pct_change": 0,
                              "volume": 0, "market_cap": 50e6, "country": "United States", "sector": "", "industry": "",
                              "exchange": "NASDAQ"} for s, d in daily.items()])
    fund = {s: {"floatShares": 8e6, "sharesOutstanding": 12e6, "shortPercentOfFloat": 0.06, "marketCap": 50e6}
            for s in daily}
    fund["DEMOM"]["floatShares"] = 3e6
    news = {"DEMOA": [{"headline": "DemoA signs supply contract with major distributor", "time": now_et.isoformat()}]}
    return universe, daily, intraday5, intraday15, fund, news
