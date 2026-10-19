"""
فحص زخم خفيف كل 5 دقائق أثناء الجلسة (بدون أسهم الارتكاز) — دقيقة تشغيل تقريباً.
    python -m faisal_scanner.quick
1) ناسداك: كل الأسهم الأمريكية بنسبة التغير والفوليوم اللحظي (3 طلبات فقط)
2) أعلى 40 سهم صاعد 5%+ يطابق الفلتر → شموع 5 دقائق + يومي من Yahoo
3) تحليل الزخم (الفواب، الوقف 2%، المقاومة القريبة) → تنبيه تيليجرام فوري لأي سهم جاهز جديد
"""
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np

from . import config as C
from . import notify
from .data_sources import ET, session_info, get_universe, YahooData, fundamentals, yahoo_quotes
from .momentum import analyze_momentum
from .pivot import READY
from .main import enrich, update_alert_state

log = logging.getLogger("faisal")


def run_quick(force=False):
    now_et = datetime.now(timezone.utc).astimezone(ET)
    session, frac = session_info(now_et)
    if session != "regular" and not force:
        log.info("quick: not regular session (%s) — skip", session)
        return None
    universe, src = get_universe()
    if src != "nasdaq-api":
        log.warning("quick: universe source is %s (no market cap) — continuing with Yahoo quotes", src)
    u = universe.copy()
    for col in ("price", "market_cap", "pct_change", "volume"):
        if col not in u.columns:
            u[col] = np.nan
    # بيانات ناسداك للتغير والفوليوم قد تكون متأخرة → نأخذ التغير والفوليوم اللحظي من Yahoo لكل الأسهم الصغيرة
    pre = u[(u["price"].isna() | ((u["price"] >= C.MIN_PRICE * 0.8) & (u["price"] <= C.MAX_PRICE * 2)))]
    pre = pre[pre["market_cap"].isna() | (pre["market_cap"] <= 0) | (pre["market_cap"] <= C.MAX_MARKET_CAP * 1.5)]
    quotes = {}
    try:
        quotes = yahoo_quotes(list(pre["symbol"]))
    except Exception as e:  # noqa
        log.warning("quick: yahoo quotes failed: %s", e)
    live = 0
    if len(quotes) >= 0.5 * len(pre):
        u = pre.copy()
        u["price"] = [quotes.get(s, {}).get("reg_price") or np.nan for s in u["symbol"]]
        u["pct_change"] = [quotes.get(s, {}).get("reg_chg") for s in u["symbol"]]
        u["volume"] = [quotes.get(s, {}).get("reg_vol") or 0 for s in u["symbol"]]
        u["pct_change"] = u["pct_change"].astype(float)
        live = 1
    elif src != "nasdaq-api":
        log.warning("quick: no live change/volume available — skip")
        return None
    else:
        u = pre
    u = u[(u["price"] >= C.MIN_PRICE) & (u["price"] <= C.MAX_PRICE * 1.5)]
    u = u[(u["pct_change"] >= C.MOM_MIN_CHANGE) & (u["volume"] >= C.MOM_MIN_VOLUME)]
    u = u[u["market_cap"].isna() | (u["market_cap"] <= 0) | (u["market_cap"] <= C.MAX_MARKET_CAP * 1.15)]
    n_movers = len(u)
    u = u.sort_values("pct_change", ascending=False).head(C.MOM_QUICK_MAX_CANDIDATES)
    cands = list(u["symbol"])
    log.info("quick: %d movers (%d analysed, live=%d, quotes=%d/%d)", n_movers, len(cands), live, len(quotes), len(pre))
    result = {"generated_at_et": now_et.isoformat(), "mode": "quick", "ready": [], "radar": [], "new_alerts": [],
              "debug": {"universe": len(universe), "prefilter": len(pre), "quotes": len(quotes), "live": live,
                        "movers": n_movers, "analysed": len(cands),
                        "top": [[r["symbol"], round(float(r["pct_change"]), 3), int(r["volume"] or 0)]
                                for r in u.head(25).to_dict("records")]}}
    if cands:
        yd = YahooData()
        daily = yd.daily(cands, period="6mo")
        intr = yd.intraday(cands, "5m", prepost=True)
        moms = []
        for s in cands:
            try:
                m = analyze_momentum(s, intr.get(s), daily.get(s), now_et, session, frac, segment="regular")
                if m:
                    moms.append(m)
            except Exception as e:  # noqa
                log.debug("quick %s: %s", s, e)
        urows = {r["symbol"]: r for r in u.to_dict("records")}
        fund = fundamentals([m.ticker for m in moms]) if moms else {}
        result["debug"]["analysed_ok"] = len(moms)
        moms = [m for m in moms if enrich(m, fund, urows.get(m.ticker, {}))]
        result["debug"]["kept"] = len(moms)
        moms.sort(key=lambda x: (x.status == READY, x.rank), reverse=True)
        result["ready"] = [m.to_dict() for m in moms if m.status == READY]
        result["radar"] = [m.to_dict() for m in moms if m.status != READY][:20]
        result["new_alerts"] = update_alert_state({"ready": result["ready"], "faisal": []}, now_et)
        for s in result["ready"]:
            if s["ticker"] in result["new_alerts"]:
                notify.send(notify.alert_text(s, now_et.astimezone(notify_tz()).strftime("%H:%M الرياض")))
    # 📌 قائمة المتابعة (أسهم فيصل + الأسهم اللي طلبت متابعتها): الوصول لمنطقة الطلب أو كسر الوقف
    try:
        watch_alerts(result, quotes, now_et)
    except Exception as e:  # noqa
        log.warning("quick watchlist failed: %s", e)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, "momentum_quick.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    hist = os.path.join(C.RESULTS_DIR, "history")
    os.makedirs(hist, exist_ok=True)
    with open(os.path.join(hist, f"{now_et.date().isoformat()}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": now_et.isoformat(), "mode": "quick", "ready": [s["ticker"] for s in result["ready"]],
                            "new": result["new_alerts"]}, ensure_ascii=False) + "\n")
    log.info("quick done: ready=%s new=%s", [s["ticker"] for s in result["ready"]], result["new_alerts"])
    return result


def watch_alerts(result, quotes, now_et):
    from . import faisal_watch
    fw = faisal_watch.load()
    if not fw:
        return
    need = [w["ticker"] for w in fw if not (quotes.get(w["ticker"]) or {}).get("reg_price")]
    if need:
        quotes = {**quotes, **yahoo_quotes(need)}
    evs = []
    for w in fw:
        px = (quotes.get(w["ticker"]) or {}).get("reg_price")
        # الاختراق يحتاج ثبات شمعتين 15 دقيقة → يتركه للفحص الساعي؛ هنا فقط المنطقة والوقف
        ev = faisal_watch.evaluate({**w, "breakout": None}, float(px) if px else None, None, "سعر لحظي")
        evs.append(ev)
    result["watchlist"] = evs
    new = update_alert_state({"ready": [], "faisal": evs}, now_et)
    result["new_alerts"] += new
    when = now_et.astimezone(notify_tz()).strftime("%H:%M الرياض")
    for ev in evs:
        if f"{ev['ticker']} ({ev['label']})" in new:
            notify.send(f"🚨 {ev['label']} — {ev['ticker']} ({when}): {ev['state_ar']} · السعر {notify.fmt_price(ev['price'])}"
                        + (f" · الوقف {ev['levels']['stop']:g}" if ev["levels"].get("stop") else ""))


def notify_tz():
    from zoneinfo import ZoneInfo
    return ZoneInfo("Asia/Riyadh")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_quick(force="--force" in sys.argv)
