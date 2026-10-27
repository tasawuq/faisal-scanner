"""
إعادة تشغيل قواعد فيصل (method.py) على بيانات حقيقية سابقة — للتحقق قبل ما توصل أي إشارة للمستخدم.
    python -m faisal_scanner.replay [days=120]

لكل سهم بمدى فيصل (ناسداك، تحت 5$، فوليوم فوق 100K، فلوت/عدد أسهم 541 ألف – 8M حسب البيانات الحالية):
- كل يوم من آخر N يوم: التحليل على الشموع المكتملة لين ذاك اليوم (مثل فحص ما بعد الإغلاق).
- إذا اكتملت شروط ص86: تُسجّل الإشارة (الطلبات، الوقف، الأهداف، الشروط)، ثم النتيجة في 10 أيام بعدها:
  هل تنفذت الطلبات؟ هل ضرب الوقف أو الهدف الأول أولاً؟ (إذا الاثنين بنفس اليوم = وقف — تقدير متحفظ)
- حدود: فريم 4 ساعات ما يُعاد (يحتاج شموع دقيقة تاريخية)، والفلوت الحالي يُستخدم للماضي.
"""
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np


from . import config as C
from . import method as M
from .data_sources import get_universe, get_price_provider, fundamentals
from .pivot import analyze_pivot, apply_gate, READY

log = logging.getLogger("faisal")
OUT = os.path.join(C.RESULTS_DIR, "replay")
HORIZON = 10


def _outcome(df, i, sig):
    lad = sig.meta["ladder"]
    orders = lad["orders"]
    stop, t1 = lad["stop"], sig.t1
    filled, fill_day, res, res_day, res_px = [], None, "لم تتنفذ", None, None
    for k in range(i + 1, min(len(df), i + 1 + HORIZON)):
        lo, hi = float(df["low"].iloc[k]), float(df["high"].iloc[k])
        new = [o for o in orders if o not in filled and lo <= o]
        if new:
            filled += new
            fill_day = fill_day or str(df.index[k])[:10]
        if filled:
            if lo <= stop:
                res, res_day, res_px = "❌ وقف", str(df.index[k])[:10], stop
                break
            if t1 and hi >= t1:
                res, res_day, res_px = "✅ هدف1", str(df.index[k])[:10], t1
                break
            res = "مفتوح"
    last = float(df["close"].iloc[min(len(df) - 1, i + HORIZON)])
    avg = float(np.mean(filled)) if filled else None
    pnl = None
    if avg:
        px = res_px if res_px else last
        pnl = px / avg - 1
    return {"filled": len(filled), "fill_day": fill_day, "result": res, "result_day": res_day,
            "pnl": None if pnl is None else round(pnl, 4), "avg_fill": avg}


def replay_one(args):
    t, df, fund, urow, days = args
    out = []
    try:
        df = df.dropna(subset=["open", "high", "low", "close"])
        n = len(df)
        last_key = None
        for i in range(max(70, n - days), n):
            win = df.iloc[max(0, i - 259):i + 1]
            sig = analyze_pivot(t, win)
            if sig is None or sig.status != READY:
                continue
            from .main import enrich
            if not enrich(sig, {t: fund}, urow, win):
                continue
            apply_gate(sig)
            if sig.status != READY:
                continue
            ch = sig.meta["p86"]["checks"]
            lad = sig.meta["ladder"]
            key = (sig.setup, round(lad["critical"], 4))
            row = {"ticker": t, "date": str(df.index[i])[:10], "setup": sig.setup, "setup_ar": sig.setup_ar,
                   "close": round(float(df["close"].iloc[i]), 4), "orders": lad["orders"], "critical": lad["critical"],
                   "stop": lad["stop"], "risk": round(lad["risk"], 4), "t1": sig.t1, "t1_label": sig.t1_label,
                   "t2": sig.t2, "rr": sig.rr, "zone": [round(x, 4) for x in sig.meta["zone"]],
                   "checks": {k: v[1] for k, v in ch.items()}, "repeat": key == last_key,
                   "float": sig.meta.get("float"), "shares": sig.meta.get("shares"), "flags": sig.flags,
                   "tags": sig.tags}
            row.update(_outcome(df, i, sig))
            out.append(row)
            last_key = key
    except Exception as e:  # noqa
        log.warning("replay %s: %s", t, e)
    return t, out


def run(days=120):
    os.makedirs(os.path.join(OUT, "bars"), exist_ok=True)
    universe, src = get_universe()
    u = universe[universe["exchange"].fillna("").str.upper() == M.EXCHANGE]
    u = u[(u["price"].isna()) | ((u["price"] >= C.MIN_PRICE) & (u["price"] <= C.MAX_PRICE * 1.2))]
    u = u[u["market_cap"].isna() | (u["market_cap"] <= 0) | (u["market_cap"] <= M.MAX_MARKET_CAP)]
    syms = list(u["symbol"])
    log.info("replay: %d nasdaq symbols under $6", len(syms))
    provider = get_price_provider()
    daily = provider.daily(syms)
    keep = []
    for s, df in daily.items():
        if df is None or len(df) < 90:
            continue
        if float(df["volume"].iloc[-21:-1].mean()) < M.MIN_AVG_VOLUME:
            continue
        keep.append(s)
    log.info("replay: %d with volume >= 100K", len(keep))
    fund = fundamentals(keep)
    rng = []
    for s in keep:
        f = fund.get(s) or {}
        k = f.get("floatShares") or f.get("sharesOutstanding")
        if k and M.MIN_FLOAT <= k <= M.MAX_FLOAT:
            rng.append(s)
    log.info("replay: %d in Faisal's share range", len(rng))
    urows = {r["symbol"]: r for r in u.to_dict("records")}
    jobs = [(s, daily[s], fund.get(s) or {}, urows.get(s, {}), days) for s in rng]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        for t, out in ex.map(replay_one, jobs, chunksize=4):
            rows += out
            if out:
                daily[t].iloc[-300:].to_csv(os.path.join(OUT, "bars", f"{t}.csv"))
    rows.sort(key=lambda r: (r["date"], r["ticker"]))
    fresh = [r for r in rows if not r["repeat"]]
    filled = [r for r in fresh if r["filled"]]
    summ = {"days": days, "symbols_in_range": len(rng), "signals": len(rows), "signals_new_setups": len(fresh),
            "filled": len(filled),
            "t1": sum(1 for r in filled if r["result"].startswith("✅")),
            "stop": sum(1 for r in filled if r["result"].startswith("❌")),
            "open": sum(1 for r in filled if r["result"] == "مفتوح"),
            "avg_pnl_filled": round(float(np.mean([r["pnl"] for r in filled if r["pnl"] is not None])), 4) if filled else None,
            "provider": provider.name, "universe_source": src}
    with open(os.path.join(OUT, "signals.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summ, "signals": rows}, fh, ensure_ascii=False, indent=1, default=str)
    log.info("replay summary: %s", summ)
    return summ


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
