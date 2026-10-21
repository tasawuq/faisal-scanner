"""
فحص زخم خفيف (بدون أسهم الارتكاز) — يستدعيه الفحص المباشر كل دقيقة من البري ماركت إلى نهاية الأفتر.
    python -m faisal_scanner.quick
1) قائمة كل الأسهم الأمريكية من ناسداك + أسعار Yahoo اللحظية (الرسمية أو البري/الأفتر حسب الجلسة)
2) أعلى الأسهم الصاعدة 5%+ المطابقة للفلتر → شموع 5 دقائق + يومي (+ فوليوم البري/الأفتر من ناسداك)
3) تحليل الزخم (الفواب، الوقف، المقاومة القريبة) → تنبيه فوري لأي سهم جاهز جديد + قائمة المتابعة
"""
import json
import logging
import os
import time as _time
from datetime import datetime, timezone

import numpy as np

from . import config as C
from . import notify
from .data_sources import (ET, session_info, get_universe, YahooData, fundamentals, yahoo_quotes, nasdaq_extended,
                           fill_missing_prices)
from .momentum import analyze_momentum, _seg
from .pivot import READY
from .main import enrich, update_alert_state

log = logging.getLogger("faisal")

# داخل الفحص المباشر (عملية واحدة طويلة): اليومي يتغير ببطء → كاش 15 دقيقة يخفف الضغط على Yahoo.
# التحليل يعتمد على الأيام السابقة فقط (+ شموع اليوم من الـ5 دقائق اللي تنسحب كل مرة).
_DAILY, _EXTVOL = {}, {}
DAILY_TTL, EXTVOL_TTL = 900, 180
PX_KEY = {"pre": ("pre_price", "pre_chg"), "post": ("post_price", "post_chg"), "regular": ("reg_price", "reg_chg")}


def _daily_cached(yd, syms, session):
    now = _time.time()
    need = [s for s in syms if s not in _DAILY or now - _DAILY[s][0] > DAILY_TTL or _DAILY[s][2] != session]
    if need:
        for s, df in (yd.daily(need, period="6mo") or {}).items():
            _DAILY[s] = (now, df, session)
    return {s: _DAILY[s][1] for s in syms if s in _DAILY}


def _ext_volume(syms, session):
    """فوليوم البري/الأفتر من ناسداك (Yahoo يعطيه صفر) — كاش 3 دقائق لكل سهم."""
    now = _time.time()
    need = [s for s in syms if s not in _EXTVOL or now - _EXTVOL[s][0] > EXTVOL_TTL or _EXTVOL[s][2] != session]
    if need:
        try:
            got = nasdaq_extended(need, session)
        except Exception as e:  # noqa
            log.warning("quick: nasdaq ext failed: %s", e)
            got = {}
        for s in need:
            _EXTVOL[s] = (now, (got.get(s) or {}).get("volume"), session)
    return {s: _EXTVOL[s][1] for s in syms if s in _EXTVOL and _EXTVOL[s][1]}


def run_quick(force=False):
    now_et = datetime.now(timezone.utc).astimezone(ET)
    session, frac = session_info(now_et)
    if session not in ("pre", "regular", "post") and not force:
        log.info("quick: market closed (%s) — skip", session)
        return None
    ext = session in ("pre", "post")
    pk, ck = PX_KEY[session if ext else "regular"]
    if not C.ENABLE_MOMENTUM:
        # الزخم موقوف (طلب المستخدم: ارتكاز فقط) → هذا الفحص المباشر يراقب مستويات فيصل فقط كل دقيقة
        result = {"generated_at_et": now_et.isoformat(), "mode": "quick", "session": session, "ready": [], "radar": [],
                  "new_alerts": [], "debug": {"momentum": "off"}}
        try:
            watch_alerts(result, {}, now_et, session)
        except Exception as e:  # noqa
            log.warning("quick watchlist failed: %s", e)
        os.makedirs(C.RESULTS_DIR, exist_ok=True)
        with open(os.path.join(C.RESULTS_DIR, "momentum_quick.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1, default=str)
        if result["new_alerts"]:
            hist = os.path.join(C.RESULTS_DIR, "history")
            os.makedirs(hist, exist_ok=True)
            with open(os.path.join(hist, f"{now_et.date().isoformat()}_live.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": now_et.isoformat(), "mode": "watch", "new": result["new_alerts"]},
                                   ensure_ascii=False) + "\n")
        return result
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
        u["price"] = [quotes.get(s, {}).get(pk) or np.nan for s in u["symbol"]]
        u["pct_change"] = [quotes.get(s, {}).get(ck) for s in u["symbol"]]
        # فوليوم البري/الأفتر غير متوفر من Yahoo → يُفحص لاحقاً من ناسداك للمرشحين فقط
        u["volume"] = 0 if ext else [quotes.get(s, {}).get("reg_vol") or 0 for s in u["symbol"]]
        u["pct_change"] = u["pct_change"].astype(float)
        live = 1
    elif ext or src != "nasdaq-api":
        log.warning("quick: no live change/volume available — skip")
        return None
    else:
        u = pre
    u = u[(u["price"] >= C.MIN_PRICE) & (u["price"] <= C.MAX_PRICE * 1.5)]
    u = u[u["pct_change"] >= C.MOM_MIN_CHANGE]
    if not ext:
        u = u[u["volume"] >= C.MOM_MIN_VOLUME]
    u = u[u["market_cap"].isna() | (u["market_cap"] <= 0) | (u["market_cap"] <= C.MAX_MARKET_CAP * 1.15)]
    n_movers = len(u)
    u = u.sort_values("pct_change", ascending=False).head(C.MOM_EXT_MAX_CANDIDATES if ext else C.MOM_QUICK_MAX_CANDIDATES)
    cands = list(u["symbol"])
    log.info("quick: %d movers (%d analysed, live=%d, quotes=%d/%d)", n_movers, len(cands), live, len(quotes), len(pre))
    result = {"generated_at_et": now_et.isoformat(), "mode": "quick", "session": session, "ready": [], "radar": [], "new_alerts": [],
              "debug": {"universe": len(universe), "prefilter": len(pre), "quotes": len(quotes), "live": live,
                        "movers": n_movers, "analysed": len(cands),
                        "top": [[r["symbol"], round(float(r["pct_change"]), 3), int(r["volume"] or 0)]
                                for r in u.head(25).to_dict("records")]}}
    if cands:
        yd = YahooData()
        daily = _daily_cached(yd, cands, session)
        intr = yd.intraday(cands, "5m", prepost=True)
        ext_vol = {}
        if ext:
            today = now_et.date()
            zero = [s for s in cands if intr.get(s) is not None and len(intr[s])
                    and float(_seg(intr[s], today, session)["volume"].sum()) == 0]
            ext_vol = _ext_volume(zero, session)
            result["debug"]["ext_volume"] = len(ext_vol)
        moms = []
        for s in cands:
            try:
                m = analyze_momentum(s, intr.get(s), daily.get(s), now_et, session, frac,
                                     segment=session if ext else "regular", ext_volume=ext_vol.get(s))
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
        result["new_alerts"] = update_alert_state({"ready": result["ready"], "faisal": []}, now_et, "_live")
        for s in result["ready"]:
            if s["ticker"] in result["new_alerts"]:
                notify.send(notify.alert_text(s, now_et.astimezone(notify_tz()).strftime("%H:%M الرياض")),
                            title=notify.alert_title(s))
    # 📌 قائمة المتابعة (أسهم فيصل + الأسهم اللي طلبت متابعتها): الوصول لمنطقة الطلب أو كسر الوقف
    try:
        watch_alerts(result, quotes, now_et, session)
    except Exception as e:  # noqa
        log.warning("quick watchlist failed: %s", e)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, "momentum_quick.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    hist = os.path.join(C.RESULTS_DIR, "history")
    os.makedirs(hist, exist_ok=True)
    with open(os.path.join(hist, f"{now_et.date().isoformat()}_live.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": now_et.isoformat(), "mode": "quick", "ready": [s["ticker"] for s in result["ready"]],
                            "new": result["new_alerts"]}, ensure_ascii=False) + "\n")
    log.info("quick done: ready=%s new=%s", [s["ticker"] for s in result["ready"]], result["new_alerts"])
    return result


def watch_alerts(result, quotes, now_et, session="regular"):
    from . import faisal_watch, weekly
    fw = faisal_watch.load()
    have = {w["ticker"] for w in fw}
    # قائمة الارتكاز (أسلوب فيصل): تنبيه لما ينزل السهم لمنطقة الطلب وتكمل شمعة تأكيد (الدخول من تحت)
    fw += [w for w in weekly.load_entries() if w["ticker"] not in have]
    if not fw:
        return
    key = PX_KEY.get(session, PX_KEY["regular"])[0]
    seg = {"pre": "بري ماركت", "post": "أفتر ماركت"}.get(session, "سعر لحظي")
    need = [w["ticker"] for w in fw if not (quotes.get(w["ticker"]) or {}).get(key)]
    if need:
        try:
            quotes = {**quotes, **yahoo_quotes(need)}
        except Exception as e:  # noqa
            log.warning("quick watchlist quotes failed: %s", e)
    bars15 = {}
    try:
        bars15 = YahooData().intraday([w["ticker"] for w in fw], "15m", prepost=True) or {}
    except Exception as e:  # noqa
        log.warning("quick watchlist 15m failed: %s", e)
    # إذا Yahoo ما رجّع السعر: ناسداك (بري/أفتر) ثم آخر شمعة 15 دقيقة — بدونها الحالة "لا توجد بيانات" ولا يطلع تنبيه
    quotes, filled = fill_missing_prices(quotes, [w["ticker"] for w in fw], session, now_et, bars15)
    if filled["nasdaq"] or filled["bars"]:
        result.setdefault("debug", {})["price_fallback"] = filled
    ext = session in ("pre", "post")
    evs = []
    for w in fw:
        t = w["ticker"]
        # في البري/الأفتر: سعر الجلسة الممتدة فقط (لا نستخدم إغلاق أمس كسعر لحظي)
        px = (quotes.get(t) or {}).get(key)
        b = bars15.get(t)
        # التنبيه بعد التأكيد فقط: شمعة 15 دقيقة خضراء داخل المنطقة، شمعتين فوق الاختراق، وفي البري/الأفتر إغلاق تحت الوقف
        zok = faisal_watch.zone_confirmed(b, w.get("demand_low"), w.get("demand_high"), now_et, session)
        hok = faisal_watch.breakout_held(b, w.get("breakout"), now_et, C.INTRADAY_HOLD_BARS, session)
        sok = faisal_watch.stop_broken(b, w.get("stop"), now_et, session) if ext else None
        rok = faisal_watch.retest_confirmed(b, w.get("breakout"), now_et, session)
        ev = faisal_watch.evaluate(w, float(px) if px else None, None, seg, hold_ok=bool(hok), zone_ok=zok, stop_ok=sok,
                                   retest_ok=rok)
        ev["session"] = session
        evs.append(ev)
    result["watchlist"] = evs
    new = update_alert_state({"ready": [], "faisal": evs}, now_et, "_live")
    result["new_alerts"] += new
    when = now_et.astimezone(notify_tz()).strftime("%H:%M الرياض")
    for ev in evs:
        if f"{ev['ticker']} ({ev['label']})" in new:
            title, body = notify.watch_alert(ev, when, seg if session != "regular" else None)
            notify.send(body, title=title)


def notify_tz():
    from zoneinfo import ZoneInfo
    return ZoneInfo("Asia/Riyadh")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_quick(force="--force" in sys.argv)
