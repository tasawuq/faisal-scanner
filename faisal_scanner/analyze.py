"""
تحليل سهم واحد (أو أكثر) بالتفصيل حسب منهج فيصل — بدون فلتر الماسح، مع كل القراءات.
    python -m faisal_scanner.analyze SDST [TICKER ...]
المخرجات: results/analysis/<TICKER>.json + daily/15m csv
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np

from . import config as C
from .data_sources import ET, session_info, YahooData, fundamentals, yahoo_quotes, merge_today_bar
from .levels import swing_lows, build_zones, count_visits, resistance_levels, candle_name
from .momentum import analyze_momentum
from .pivot import Ctx, analyze_pivot, find_support, detect_w
from .main import enrich, apply_news, refresh_status

log = logging.getLogger("faisal")
OUT = os.path.join(C.RESULTS_DIR, "analysis")


def _f(x, nd=4):
    try:
        x = float(x)
        return round(x, nd) if np.isfinite(x) else None
    except Exception:  # noqa
        return None


def diagnose(t, df):
    ctx = Ctx(df)
    n = ctx.n
    start = max(0, n - C.LOOKBACK_LEVELS)
    pts = [(i, ctx.l[i]) for i in swing_lows(ctx.l, k=C.SWING_K) if start <= i < n - 3]
    zones = build_zones(pts, C.ZONE_TOL) if pts else []
    period_min = float(np.min(ctx.l[start:n - 3])) if n > 3 else float("nan")
    zl = []
    for z in zones:
        z.touches = max(z.touches, count_visits(ctx.l, ctx.c, z, start, n - 3, C.ZONE_TOL))
        zl.append({"low": _f(z.low), "high": _f(z.high), "touches": z.touches,
                   "is_bottom": bool(z.low <= period_min * 1.001),
                   "first": str(df.index[z.first_idx].date()), "last": str(df.index[z.last_idx].date())})
    zl.sort(key=lambda z: -z["high"])
    sup = find_support(ctx)
    w = detect_w(ctx)
    ma_vals = {"MA20": ctx.ma[20][-1], "MA30": ctx.ma[30][-1], "MA50": ctx.ma[50][-1]}
    res = resistance_levels(df, ctx.price, C.LOOKBACK_LEVELS, C.SWING_K, ma_vals)
    distinct = len(set(np.round(ctx.c[-20:], 4)))
    last = []
    for j in range(max(0, n - 12), n):
        last.append({"d": str(df.index[j].date()), "o": _f(ctx.o[j]), "h": _f(ctx.h[j]), "l": _f(ctx.l[j]),
                     "c": _f(ctx.c[j]), "v": int(ctx.v[j]), "rsi": _f(ctx.rsi[j], 1), "hist": _f(ctx.hist[j], 5)})
    return {
        "price": _f(ctx.price), "chg_1d": _f(ctx.c[-1] / ctx.c[-2] - 1, 4),
        "ma": {k: _f(v) for k, v in ma_vals.items()}, "ma_state": ctx.ma_state,
        "rsi_now": _f(ctx.rsi[-1], 1), "rsi_min15": _f(ctx.rsi_min(C.RSI_LOOKBACK_BOUNCE), 1),
        "rsi_min30": _f(ctx.rsi_min(C.RSI_LOOKBACK_BREAKOUT), 1),
        "macd": {"line": _f(ctx.macd_line[-1], 5), "signal": _f(ctx.macd_sig[-1], 5),
                 "hist_last3": [_f(x, 5) for x in ctx.hist[-3:]], "state": ctx.macd_state},
        "candles": [{"d": str(df.index[j].date()), **{k: v for k, v in info.items() if not isinstance(v, (np.ndarray,))}}
                    for j, info in ctx.candles],
        "atr": _f(ctx.atr[-1]), "avg_vol20": int(ctx.avg_vol), "last_vol": int(ctx.v[-1]),
        "drawdown_90d": _f(ctx.drawdown, 3), "high_90d": _f(np.nanmax(ctx.h[-90:])),
        "low_10d": _f(ctx.recent_low10), "gain_from_low10": _f(ctx.gain_from_low10, 3),
        "sell_drying": ctx.sell_drying, "vol_improving": ctx.vol_improving, "higher_lows": bool(ctx.higher_lows),
        "pump_recent": ctx.pump_recent, "distribution_recent": ctx.distribution_recent,
        "pump_month": None if not ctx.pump_month else {"d": str(df.index[ctx.pump_month[0]].date()),
                                                       "high": _f(ctx.pump_month[1]), "close": _f(ctx.pump_month[2]),
                                                       "vol_mult": _f(ctx.pump_month[3], 1)},
        "former_runner": ctx.former_runner, "runner_pct": _f(ctx.runner_pct, 3),
        "distinct_closes20": distinct,
        "support_zones": zl[:8],
        "support_selected": None if not sup else {"low": _f(sup[0].low), "high": _f(sup[0].high),
                                                  "touches": sup[0].touches, "interacting": sup[1]},
        "w": None if not w else {"low1": _f(w["low1"]), "low2": _f(w["low2"]), "neck": _f(w["neck"]),
                                 "low1_d": str(df.index[w["a"]].date()), "low2_d": str(df.index[w["b"]].date())},
        "resistances": [[_f(a), b] for a, b in res[:6]],
        "high_52w": _f(np.nanmax(ctx.h[-250:])), "low_52w": _f(np.nanmin(ctx.l[-250:])),
        "last_bars": last,
    }


def level_map(df, i15, price):
    """خريطة المستويات القريبة فوق وتحت السعر — تشمل اللي يفوّتها تحليل القمم والقيعان اليومي:
    قمة/قاع آخر الأيام، فواب الجلسة الأخيرة، قاعدة آخر ساعة، إغلاقات متجمعة، دعم مكسور صار مقاومة، المتوسطات."""
    from datetime import time as _t
    lv = []
    n = len(df)
    for j in range(max(0, n - 5), n):
        d = str(df.index[j].date())
        lv.append((float(df["high"].iloc[j]), f"قمة يوم {d}"))
        lv.append((float(df["low"].iloc[j]), f"قاع يوم {d}"))
    for p_ in (20, 30, 50):
        if n >= p_:
            lv.append((float(df["close"].iloc[-p_:].mean()), f"MA{p_}"))
    # إغلاقات/قيعان سابقة (آخر 60 يوم) — الدعم المكسور يصير مقاومة والعكس
    closes = df["close"].values[-60:-1]
    lows = df["low"].values[-60:-1]
    for arr, lab in ((closes, "إغلاقات سابقة"), (lows, "قيعان سابقة")):
        a = np.sort(arr)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[j + 1] <= a[i] * 1.03:
                j += 1
            if j - i + 1 >= 3:
                lv.append((float(np.median(a[i:j + 1])), f"{lab} متجمعة ({j - i + 1})"))
            i = j + 1
    for lvl, lab in resistance_levels(df, price, 120, 3, {}, floor_pct=0.0):
        lv.append((lvl, lab))
    if i15 is not None and len(i15):
        b = i15.copy()
        days = sorted({ts.date() for ts in b.index if _t(9, 30) <= ts.time() < _t(16, 0)})
        if days:
            last = days[-1]
            reg = b[[ts.date() == last and _t(9, 30) <= ts.time() < _t(16, 0) for ts in b.index]]
            if len(reg) and reg["volume"].sum() > 0:
                tp = (reg["high"] + reg["low"] + reg["close"]) / 3
                lv.append((float((tp * reg["volume"]).sum() / reg["volume"].sum()), f"فواب جلسة {last}"))
                lh = reg[[ts.time() >= _t(15, 0) for ts in reg.index]]
                if len(lh):
                    lv.append((float(lh["low"].min()), f"قاعدة آخر ساعة {last}"))
            ext = b[[ts.date() >= last and not (_t(9, 30) <= ts.time() < _t(16, 0)) and
                     (ts.date() > last or ts.time() >= _t(16, 0)) for ts in b.index]]
            if len(ext):
                lv.append((float(ext["high"].max()), "قمة الأفتر/البري"))
                lv.append((float(ext["low"].min()), "قاع الأفتر/البري"))
    above = sorted([x for x in lv if x[0] > price * 1.002], key=lambda x: x[0])
    below = sorted([x for x in lv if x[0] < price * 0.998], key=lambda x: -x[0])

    def merge(items):
        out = []
        for v, lab in items:
            if out and abs(v / out[-1][0] - 1) <= 0.02:
                if lab not in out[-1][1]:
                    out[-1] = (out[-1][0], out[-1][1] + " + " + lab, out[-1][2] + 1)
                continue
            out.append((v, lab, 1))
        return [{"level": _f(v), "why": lab, "confluence": c} for v, lab, c in out]
    at = [x for x in lv if abs(x[0] / price - 1) <= 0.002]
    return {"price": _f(price), "at_price": [lab for _, lab in at], "above": merge(above)[:8], "below": merge(below)[:8]}


def run(tickers):
    now_et = datetime.now(timezone.utc).astimezone(ET)
    session, frac = session_info(now_et)
    yd = YahooData()
    daily = yd.daily(tickers, period="2y")
    i15 = yd.intraday(tickers, "15m", prepost=True)
    i5 = yd.intraday(tickers, "5m", prepost=True)
    fund = fundamentals(tickers)
    try:
        quotes = yahoo_quotes(tickers)
    except Exception as e:  # noqa
        quotes = {}
        log.warning("quotes: %s", e)
    try:
        news = yd.news(tickers, days=30)
    except Exception:  # noqa
        news = {}
    os.makedirs(OUT, exist_ok=True)
    today = now_et.date()
    for t in tickers:
        df = daily.get(t)
        rep = {"ticker": t, "generated_at_et": now_et.isoformat(), "session": session,
               "quote": quotes.get(t), "news": (news.get(t) or [])[:12]}
        f = fund.get(t) or {}
        rep["fund"] = {k: f.get(k) for k in ("floatShares", "sharesOutstanding", "marketCap", "shortPercentOfFloat",
                                             "longName", "country", "industry", "sector", "lastSplitFactor",
                                             "lastSplitDate", "averageVolume", "heldPercentInsiders",
                                             "heldPercentInstitutions", "totalCash", "totalDebt", "fiftyTwoWeekHigh",
                                             "fiftyTwoWeekLow") if k in f}
        if df is None or len(df) < 60:
            rep["error"] = "no daily data"
        else:
            if t in i15 and session == "regular":
                df, _ = merge_today_bar(df, i15[t], today)
            rep["daily"] = diagnose(t, df)
            q = quotes.get(t) or {}
            live = q.get("pre_price") or q.get("post_price") or q.get("reg_price") or rep["daily"]["price"]
            try:
                rep["levels_map"] = level_map(df, i15.get(t), float(live))
            except Exception as e:  # noqa
                rep["levels_map"] = {"error": str(e)[:200]}
            df.iloc[-260:].to_csv(os.path.join(OUT, f"{t}_daily.csv"))
            partial = session == "regular" and df.index[-1].date() == today
            sig = analyze_pivot(t, df, frac, partial)
            if sig:
                enrich(sig, fund, {})
                apply_news(sig, news.get(t))
                refresh_status(sig)
                rep["pivot"] = sig.to_dict()
            else:
                rep["pivot"] = None
            seg = session if session in ("pre", "regular", "post") else "regular"
            try:
                m = analyze_momentum(t, i5.get(t), df, now_et, session, frac, segment=seg)
                rep["momentum"] = m.to_dict() if m else None
            except Exception as e:  # noqa
                rep["momentum"] = {"error": str(e)[:200]}
        if t in i15:
            i15[t].iloc[-200:].to_csv(os.path.join(OUT, f"{t}_15m.csv"))
        with open(os.path.join(OUT, f"{t}.json"), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
        log.info("analysed %s", t)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = [a.strip().upper() for a in " ".join(sys.argv[1:]).replace(",", " ").split() if a.strip()]
    run(args or ["SDST"])
