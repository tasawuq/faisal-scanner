"""
تشغيل ماسح منهج فيصل.

    python -m faisal_scanner.main            # فحص واحد (يتخطى إذا السوق مغلق)
    python -m faisal_scanner.main --force    # فحص حتى لو السوق مغلق
    python -m faisal_scanner.main --loop 60  # فحص كل 60 دقيقة على جهازك
    python -m faisal_scanner.main --demo     # تجربة على بيانات اصطناعية بدون إنترنت
"""
import argparse
import json
import logging
import os
import sys
import time as _time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from . import config as C
from .data_sources import (ET, session_info, get_universe, get_price_provider, YahooData, AlpacaData,
                           merge_today_bar, fundamentals, yahoo_quotes, load_daily_cache, save_daily_cache,
                           nasdaq_extended, fill_missing_prices)
from .pivot import analyze_pivot, apply_gate, READY, WATCH
from . import method as M
from .momentum import analyze_momentum
from .report import build_markdown
from . import faisal_watch

log = logging.getLogger("faisal")


# ------------------------------------------------------------------ helpers
def last_split(f):
    """آخر تقسيم (أي تاريخ) بصيغة «عكسي 1:10 (2026-09-28)» — ص56: فيصل يكتب آخر تقسيم لكل سهم."""
    info = M.split_info(f)
    if not info:
        return None
    return f"{'عكسي' if info['reverse'] else 'عادي'} {info['factor']} ({info['date']})"


def _m(x):
    x = float(x)
    return f"{x / 1e6:.2f}M" if x < 10e6 else f"{x / 1e6:.1f}M"


def data_line(meta: dict) -> str:
    """قالب فيصل (ص32، ص38، ص90–94): الجنسية - النشاط - التقسيم (تاريخه، الافتتاح بعده، الأعلى بعده) - الشورت المتاح -
    الفلووت/عدد الأسهم - السلوك السابق. (الأخبار بتواريخها في سطر مستقل.)"""
    parts = []
    if meta.get("country_ar"):
        parts.append(meta["country_ar"])
    if meta.get("industry"):
        parts.append(meta["industry"])
    sp = meta.get("split") or {}
    if sp:
        t = f"{'قسم' if sp.get('reverse') else 'تقسيم عادي'} {sp['factor']} ({sp['date']})"
        if sp.get("open_after") is not None:
            t += f" افتتاح {M.rnd(sp['open_after'])} - اعلى {M.rnd(sp['high_after'])}"
        parts.append(t)
    elif meta.get("last_split"):
        parts.append(f"آخر تقسيم: {meta['last_split']}")
    else:
        parts.append("بدون تقسيم مسجل")
    parts.append("الشورت المتاح: " + (f"{int(meta['short_available']):,}" if meta.get("short_available") is not None
                                      else "غير متوفر"))
    flt, sh = meta.get("float"), meta.get("shares")
    parts.append(f"فلوت {_m(flt)}" if flt else "فلوت غير معروف")
    if sh:
        parts.append(f"عدد الأسهم {_m(sh)}")
    run = meta.get("runner") or {}
    if run.get("is_runner"):
        parts.append(f"سبق صعد من {M.rnd(run['from'])} ({run['from_date']}) إلى {M.rnd(run['to'])} ({run['to_date']}) "
                     f"+{run['pct'] * 100:.0f}%")
    return " - ".join(parts)


def news_line(items) -> str:
    """الأخبار بتواريخها (قالب فيصل ص90–94: «اخباره : … 15/5 - … 22/4»)."""
    out = []
    for it in (items or [])[:3]:
        if isinstance(it, dict):
            t, ts = it.get("headline", ""), (it.get("time") or "")[:10]
        else:
            ts, t = it[0], it[1]
        if t:
            out.append(f"{t} ({ts})" if ts else t)
    return " | ".join(out)


def enrich(sig, fund: dict, urow: dict, df=None) -> bool:
    """فلتر فيصل (ص6، ص61) + بيانات قالبه. يرجع False إذا السهم خارج الفلتر."""
    f = fund.get(sig.ticker) or {}
    flt = f.get("floatShares") or None
    sh = f.get("sharesOutstanding") or None
    if flt and flt < 50_000:  # بيانات غير منطقية
        flt = None
    mcap = f.get("marketCap") or urow.get("market_cap")
    sig.meta["float"] = flt or None
    sig.meta["shares"] = sh
    sig.meta["market_cap"] = mcap if mcap and not (isinstance(mcap, float) and np.isnan(mcap)) else None
    sig.meta["exchange"] = urow.get("exchange", "")
    sig.meta["name"] = f.get("longName") or urow.get("name", "")
    known = flt or (sh if sh and sh <= M.MAX_FLOAT else None)
    if known and known > M.MAX_FLOAT:
        return False  # ص61: Float Under 20M
    if sig.meta["market_cap"] and sig.meta["market_cap"] > M.MAX_MARKET_CAP:
        return False  # ص6: Under 300M
    if not known:
        sig.flags.append("الفلووت غير معروف — تحقق يدوياً قبل أي دخول (فلتر ص61 تحت 20M)")
    elif known < 5e6:
        sig.tags.insert(0, f"فلووت {_m(known)} 🔥 (ص8: أقل من 5M قوي وخطر)")
    country = (f.get("country") or urow.get("country") or "")
    sig.meta["country_ar"] = M.country_ar(country)
    sig.meta["industry"] = f.get("industry") or urow.get("industry") or ""
    ind = (sig.meta["industry"] + " " + (urow.get("industry") or "")).lower()
    if "china" in country.lower() or "hong kong" in country.lower():
        sig.flags.append("سهم صيني عالي التذبذب")
    if any(w in ind for w in ("biotech", "pharma", "medical", "drug", "diagnostic")) and "retail" not in ind:
        sig.flags.append("سهم طبي — الخبر قد يغير الاتجاه (ص72، ص94)")
    sp = M.split_info(f, df)
    sig.meta["split"] = sp
    sig.meta["last_split"] = last_split(f)
    if sp.get("recent"):
        t = f"تقسيم عكسي حديث {sp['factor']} ({sp['date']})"
        if sp.get("open_after") is not None:
            t += f" — افتتح بعده {M.rnd(sp['open_after'])} وأعلى سعر {M.rnd(sp['high_after'])}"
        sig.tags.insert(0, t)
    sig.meta["short_available"] = None  # فيصل يكتب الشورت المتاح بعدد الأسهم (ص8، ص56) — ما عندنا مصدر له بعد
    if f.get("shortPercentOfFloat"):
        sig.meta["short_pct_float"] = f["shortPercentOfFloat"]
    if sig.price < 0.5:
        sig.flags.append("سعر منخفض جداً — سبريد واسع")
    return True


def refresh_status(sig):
    if sig.kind == "pivot":
        apply_gate(sig)
        return
    sig.status = READY if not sig.waiting_for and sig.setup not in (
        "MOM_ABOVE_VWAP", "MOM_EXTENDED", "MOM_BELOW_VWAP") else WATCH


def apply_news(sig, items):
    if not items:
        return
    sig.news = items[:3]
    text = " ".join(i["headline"].lower() for i in items)
    if any(w in text for w in C.OFFERING_WORDS):
        sig.flags.append("خبر طرح/تخفيف حديث (Offering Trap — ص10)")
        if sig.kind == "pivot":
            sig.meta.setdefault("static_wait", []).append("خبر طرح أسهم/تخفيف — «خبر سلبي يلغي النموذج غالباً» (ص67، ص89)")
        else:
            sig.waiting_for.append("خبر طرح أسهم/تخفيف — إدارة الخطر أولاً")
    elif any(w in text for w in C.CATALYST_WORDS):
        sig.tags.append("محفز: خبر حديث")
    refresh_status(sig)


def upgrade_4h(sig, bars15, now_et, delay_min: int) -> bool:
    """ص90 «قراءة الشموع تتم ع جميع الفريمات / اقربها 4 ساعات»: إذا شروط ص86 ما اكتملت على اليومي المكتمل،
    نقرأ آخر شمعة 4 ساعات مكتملة اليوم (9:30–13:30 أو 13:30–16:00). الترقية فقط إذا اكتملت كل الشروط."""
    p = sig.meta.get("p86")
    if sig.kind != "pivot" or not p or p.get("complete") or sig.meta.get("breakout_setup") or bars15 is None:
        return False
    z_low, z_high = sig.meta["zone"]
    r = M.p86_4h(p, bars15, z_low, z_high, sig.meta.get("last_close"), sig.meta.get("avg_volume") or 0, now_et,
                 delay_min)
    if not r or not r["complete"]:
        return False
    keep = {k: p[k] for k in ("below_ma", "broke", "episodes", "breakout") if k in p}
    sig.meta["p86"] = {**keep, **{k: (v if k != "checks" else {kk: [bool(x[0]), x[1]] for kk, x in v.items()})
                                  for k, v in r.items()}}
    if sig.setup in ("BROKE_SUPPORT", "AT_SUPPORT"):
        sig.setup = "SWEEP" if "سحب" in r["checks"]["retest"][1] else "BOUNCE"
        from .pivot import SETUP_AR
        sig.setup_ar = SETUP_AR[sig.setup]
    apply_gate(sig)
    return True


# ------------------------------------------------------------------ run
def run_once(force=False, demo=False, limit=None) -> dict:
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    session, frac = session_info(now_et)
    errors = []
    if demo:
        from .demo import demo_data
        universe, daily, intraday5, intraday15, fund, news = demo_data(now_et)
        provider_name = "demo"
        session, frac = "post", 1.0  # بعد الإغلاق: شمعة اليوم مكتملة (التحليل على الشموع المكتملة فقط)
        intra_provider = None
        quotes = {}
    else:
        if session == "closed" and not force:
            log.info("Market closed (%s ET) — skipping", now_et.strftime("%H:%M"))
            return {}
        universe, usrc = get_universe()
        log.info("Universe: %d symbols from %s", len(universe), usrc)
        has_px = universe["price"].notna().any()
        u = universe
        # فلتر فيصل (ص6، ص7، ص61): ناسداك فقط، السعر تحت 5$، القيمة السوقية تحت 300M
        n_all = len(u)
        if "exchange" in u.columns:
            u = u[u["exchange"].fillna("").str.upper() == M.EXCHANGE]
        n_nasdaq = len(u)
        if has_px:
            u = u[(u["price"] >= C.MIN_PRICE) & (u["price"] <= C.MAX_PRICE * 1.2)]
            u = u[u["market_cap"].isna() | (u["market_cap"] <= 0) | (u["market_cap"] <= M.MAX_MARKET_CAP * 1.15)]
        syms = list(u["symbol"])[:limit] if limit else list(u["symbol"])
        # أسهم فيصل تُفحص دائماً حتى لو خارج الفلتر
        have = set(syms)
        syms += [w["ticker"] for w in faisal_watch.load() if w["ticker"] not in have]
        log.info("Pre-filtered (price/cap): %d", len(syms))
        provider = get_price_provider()
        provider_name = provider.name
        # في البري والأفتر الشموع اليومية لا تتغير خلال الجلسة → نعيد استخدامها من الذاكرة
        cache_key = f"{now_et.date().isoformat()}_{session}" if session in ("pre", "post") else None
        if session == "closed" and now_et.hour >= 20:
            # بعد 8 مساءً نيويورك Yahoo أحياناً يحذف شمعة اليوم مؤقتاً → نستخدم شموع الأفتر ماركت المحفوظة
            cache_key = f"{now_et.date().isoformat()}_post"
        daily = load_daily_cache(cache_key) if cache_key else None
        if daily:
            provider_name += " (cached daily)"
            log.info("Daily bars from cache %s (%d)", cache_key, len(daily))
        else:
            try:
                daily = provider.daily(syms)
            except Exception as e:  # noqa
                errors.append(f"{provider.name} daily failed: {str(e)[:120]}")
                log.warning("primary daily failed (%s) — Yahoo fallback", e)
                provider, provider_name = YahooData(), "yahoo (fallback)"
                daily = provider.daily(syms)
        intraday15 = {}
        if isinstance(provider, AlpacaData) and session in ("regular", "post"):
            try:
                intraday15 = provider.intraday(syms, "15m")
            except Exception as e:  # noqa
                errors.append(f"alpaca intraday: {str(e)[:100]}")
            today = now_et.date()
            urows = universe.set_index("symbol")
            for s in list(daily):
                if s in intraday15:
                    daily[s], _ = merge_today_bar(daily[s], intraday15[s], today)
                    px = urows["price"].get(s) if s in urows.index else None
                    if session == "regular" and px and np.isfinite(px) and daily[s].index[-1].date() == today:
                        i = daily[s].index[-1]
                        daily[s].loc[i, "close"] = px
                        daily[s].loc[i, "high"] = max(daily[s].loc[i, "high"], px)
                        daily[s].loc[i, "low"] = min(daily[s].loc[i, "low"], px)
            if cache_key and daily and session != "closed":
                try:
                    save_daily_cache(cache_key, daily)
                except Exception as e:  # noqa
                    log.warning("daily cache save failed: %s", e)
        intra_provider = provider
        fund, news, intraday5 = None, None, None
        quotes = {}
        # الزخم يحتاج أسعار البري/الأفتر لكل الأسهم؛ الارتكاز فقط → نطلبها بعد التحليل للمرشحين فقط (طلبات أقل = حد Yahoo أبعد)
        if session in ("pre", "post") and C.ENABLE_MOMENTUM:
            try:
                quotes = yahoo_quotes(syms)
                log.info("Extended-hours quotes: %d", len(quotes))
            except Exception as e:  # noqa
                errors.append(f"quotes: {str(e)[:80]}")

    urows = {r["symbol"]: r for r in universe.to_dict("records")}
    today = now_et.date()
    # شمعة اليوم تُعتبر مكتملة بعد الإغلاق + تأخير بيانات SIP (آخر شمعة 15 دقيقة توصل متأخرة) — قبلها تُعامل كجارية
    close_dt = datetime.combine(today, datetime.min.time(), tzinfo=ET).replace(hour=16)
    day_closed = demo or now_et >= close_dt + timedelta(minutes=C.ALPACA_SIP_DELAY_MIN + 5)

    # ---------------- pivot analysis
    pivots = []
    analyzed = 0
    funnel = {"prefiltered": len(syms) if not demo else len(daily), "with_data": len(daily), "short_history": 0,
              "price_out": 0, "low_volume": 0, "outside_float": 0,
              "nasdaq": None if demo else n_nasdaq, "all": None if demo else n_all}
    for s, df in daily.items():
        try:
            if df is None or len(df) < 60:
                funnel["short_history"] += 1  # أقل من 60 يوم تداول (طرح جديد — ص69 يحتاج مستويات هندسية)
                continue
            avg = float(df["volume"].iloc[-21:-1].mean())
            px = float(df["close"].iloc[-1])
            if not (C.MIN_PRICE <= px <= C.MAX_PRICE):
                funnel["price_out"] += 1
                continue
            if avg < M.MIN_AVG_VOLUME:  # ص6، ص61: Average Volume Over 100K
                funnel["low_volume"] += 1
                continue
            analyzed += 1
            partial = df.index[-1].date() == today and (session == "regular" or (session == "post" and not day_closed))
            sig = analyze_pivot(s, df, frac, partial)
            if sig:
                pivots.append(sig)
        except Exception as e:  # noqa
            log.debug("pivot %s failed: %s", s, e)

    # ---------------- أسعار البري/الأفتر لمرشحي الارتكاز + قائمة المتابعة (Yahoo ثم ناسداك احتياط)
    if session in ("pre", "post") and not demo:
        pk = "pre_price" if session == "pre" else "post_price"
        top_p = sorted(pivots, key=lambda x: (x.status != READY, -x.rank))[:60]
        want = list(dict.fromkeys([x.ticker for x in top_p] + [w["ticker"] for w in faisal_watch.load()]))
        need = [s for s in want if not (quotes.get(s) or {}).get(pk)]
        yq = {}
        if need:
            try:
                yq = yahoo_quotes(need)
                quotes = {**quotes, **yq}
            except Exception as e:  # noqa
                errors.append(f"quotes: {str(e)[:80]}")
        n_y = sum(1 for s in want if (quotes.get(s) or {}).get(pk))
        quotes, filled = fill_missing_prices(quotes, want, session)
        log.info("Extended prices: yahoo=%d nasdaq=%d of %d", n_y, filled["nasdaq"], len(want))
        if need and not yq:
            errors.append(f"yahoo quotes empty in {session} — nasdaq fallback {filled['nasdaq']}/{len(want)}")

    # ---------------- momentum candidates
    moms = []
    ext_movers = 0
    ext_debug = []
    if C.ENABLE_MOMENTUM and session in ("pre", "regular", "post"):
        if session == "regular":
            cands = set()
            for s, df in daily.items():
                if df is None or len(df) < 21 or df.index[-1].date() != today:
                    continue
                prev = float(df["close"].iloc[-2])
                chg = float(df["close"].iloc[-1]) / prev - 1
                gap = float(df["open"].iloc[-1]) / prev - 1
                if max(chg, gap) >= C.MOM_MIN_CHANGE and float(df["volume"].iloc[-1]) >= C.MOM_MIN_VOLUME:
                    cands.add(s)
            cands = sorted(cands, key=lambda s: -(daily[s]["close"].iloc[-1] / daily[s]["close"].iloc[-2]))[:60]
        else:
            # أسهم تتحرك في البري/الأفتر ماركت (خبر، قروب، فوليوم)
            pk, ck = ("pre_price", "pre_chg") if session == "pre" else ("post_price", "post_chg")
            movers = []
            for s, q in quotes.items():
                px, ch = q.get(pk), q.get(ck) or 0
                if s in daily and px and ch >= C.MOM_MIN_CHANGE and C.MIN_PRICE <= px <= C.MAX_PRICE * 1.5:
                    movers.append((ch, s))
            cands = [s for _, s in sorted(movers, reverse=True)[:C.MOM_EXT_MAX_CANDIDATES]]
            ext_movers = len(movers)
        if cands:
            if intraday5 is None:
                intraday5 = {}
                # في البري/الأفتر: Alpaca أولاً لأنه يعطي فوليوم الجلسة الممتدة (Yahoo يعطيه صفر)
                if session in ("pre", "post") and isinstance(intra_provider, AlpacaData):
                    try:
                        intraday5 = intra_provider.intraday(cands, "5m", prepost=True)
                    except Exception as e:  # noqa
                        errors.append(f"alpaca ext 5m: {str(e)[:80]}")
                try:
                    need = [c for c in cands if c not in intraday5]
                    if need:
                        intraday5.update(YahooData().intraday(need, "5m", prepost=True))
                except Exception as e:  # noqa
                    errors.append(f"yahoo 5m: {str(e)[:80]}")
                missing = [c for c in cands if c not in intraday5]
                if missing and isinstance(intra_provider, AlpacaData):
                    try:
                        intraday5.update(intra_provider.intraday(missing, "5m", prepost=True))
                    except Exception as e:  # noqa
                        errors.append(f"alpaca 5m: {str(e)[:80]}")
            ext_vol = {}
            if session in ("pre", "post") and cands:
                from .momentum import _seg as _segbars
                zero = []
                for c_ in cands:
                    b_ = intraday5.get(c_)
                    sb_ = _segbars(b_, today, session) if b_ is not None and len(b_) else None
                    if sb_ is None or len(sb_) == 0 or float(sb_["volume"].sum()) == 0:
                        zero.append(c_)
                if zero:
                    try:
                        nx = nasdaq_extended(zero, "pre" if session == "pre" else "post",
                                             raw_dump=os.path.join(C.RESULTS_DIR, "debug", "nasdaq_ext_sample.json"))
                        ext_vol = {k: v.get("volume") for k, v in nx.items() if v.get("volume")}
                    except Exception as e:  # noqa
                        errors.append(f"nasdaq ext: {str(e)[:80]}")
            for s in cands:
                if session in ("pre", "post") and len(ext_debug) < 15:
                    from .momentum import _seg
                    intr = intraday5.get(s)
                    sb = _seg(intr, today, session) if intr is not None and len(intr) else None
                    q = quotes.get(s, {})
                    ext_debug.append({"t": s, "chg": round(q.get("pre_chg" if session == "pre" else "post_chg") or 0, 3),
                                      "bars": 0 if sb is None else len(sb),
                                      "vol": 0 if sb is None else float(sb["volume"].sum())})
                try:
                    m = analyze_momentum(s, intraday5.get(s), daily[s], now_et, session, frac, segment=session,
                                         ext_volume=ext_vol.get(s))
                    if m:
                        moms.append(m)
                except Exception as e:  # noqa
                    log.debug("momentum %s failed: %s", s, e)

    # ---------------- fundamentals (float / short / split) for the shortlist
    pivots.sort(key=lambda x: (x.status == READY, x.rank), reverse=True)
    pivots = pivots[:150]
    moms.sort(key=lambda x: (x.status == READY, x.rank), reverse=True)
    moms = moms[:30]
    tick = sorted({x.ticker for x in pivots + moms})
    if fund is None:
        try:
            fund = fundamentals(tick)
        except Exception as e:  # noqa
            fund = {}
            errors.append(f"fundamentals: {str(e)[:80]}")
    kept = []
    for sig in pivots:
        if not enrich(sig, fund, urows.get(sig.ticker, {}), daily.get(sig.ticker)):
            funnel["outside_float"] += 1
            continue
        apply_gate(sig)
        kept.append(sig)
    pivots = kept
    if quotes:
        seg_ar = "البري" if session == "pre" else "الأفتر"
        pk, ck = ("pre_price", "pre_chg") if session == "pre" else ("post_price", "post_chg")
        for sig in pivots:
            q = quotes.get(sig.ticker) or {}
            px, ch = q.get(pk), q.get(ck)
            if not px:
                continue
            sig.meta["ext_price"], sig.meta["ext_chg"], sig.meta["ext_session"] = px, ch, session
            if sig.stop and px < sig.stop:
                sig.flags.append(f"نزل في {seg_ar} تحت الوقف ({px:.3g})")
                sig.meta.setdefault("static_wait", []).append(f"السعر في {seg_ar} {px:.3g} تحت الوقف — الدعم مكسور حالياً")
                refresh_status(sig)
    moms = [m for m in moms if enrich(m, fund, urows.get(m.ticker, {}))]
    for m in moms:
        if m.meta.get("float") is None:
            m.tags.append("تحقق من الفلووت")

    # ---------------- news for the top of the list
    top = [x for x in pivots if x.status == READY] + [x for x in pivots if x.status == WATCH][:25] + moms[:10]
    if news is None:
        try:
            src = intra_provider if isinstance(intra_provider, AlpacaData) else YahooData()
            news = src.news([x.ticker for x in top])
        except Exception as e:  # noqa
            news = {}
            errors.append(f"news: {str(e)[:80]}")
    for x in top:
        x.news = []
        apply_news(x, news.get(x.ticker))

    # ---------------- فريم 4 ساعات (ص90 «اقربها 4 ساعات»): شمعة 4 ساعات مكتملة اليوم تكمل شروط ص86
    if session in ("regular", "post") and intraday15 and not demo:
        delay = C.ALPACA_SIP_DELAY_MIN if isinstance(intra_provider, AlpacaData) else 0
        for x in pivots:
            try:
                if upgrade_4h(x, intraday15.get(x.ticker), now_et, delay):
                    x.tags.append("اكتملت الشروط على فريم 4 ساعات")
            except Exception as e:  # noqa
                errors.append(f"4h {x.ticker}: {str(e)[:60]}")

    for x in pivots + moms:
        if not hasattr(x, "news"):
            x.news = []
    # ---------------- 📋 قائمة الارتكاز بأسلوب فيصل (ص70–73، ص54، ص59) — ثابتة خلال الجلسة، تتجدد كل 3 ساعات
    if not demo and (session in ("pre", "regular", "post") or force):
        try:
            from . import weekly
            if force or weekly.due(now_et):
                weekly.build(pivots, daily, fund, now_et)
        except Exception as e:  # noqa
            errors.append(f"weekly: {str(e)[:80]}")

    ready = sorted([x for x in pivots + moms if x.status == READY], key=lambda x: -x.rank)
    watch = sorted([x for x in pivots if x.status == WATCH], key=lambda x: -x.rank)
    momw = sorted([x for x in moms if x.status == WATCH], key=lambda x: -x.rank)

    def dump(x):
        d = x.to_dict()
        d["news"] = getattr(x, "news", [])
        return d

    # ---------------- 📌 أسهم فيصل (مستوياته المنشورة)
    faisal = []
    fw = faisal_watch.load()
    if fw:
        by_t = {x.ticker: x for x in pivots}
        brk_bars = {}
        if session in ("pre", "regular", "post") and not demo:
            need = [w["ticker"] for w in fw]
            if need:
                try:
                    brk_bars = YahooData().intraday(need, "15m", prepost=True)
                except Exception as e:  # noqa
                    errors.append(f"faisal 15m: {str(e)[:60]}")
            if session in ("pre", "post") and brk_bars:
                quotes, _ = fill_missing_prices(quotes, [w["ticker"] for w in fw], session, now_et, brk_bars, max_nasdaq=0)
        for w in fw:
            t = w["ticker"]
            df = daily.get(t)
            last_close = float(df["close"].iloc[-1]) if df is not None and len(df) else None
            px, label = last_close, "آخر سعر"
            q = quotes.get(t) or {}
            if session in ("pre", "post"):
                ep = q.get("pre_price" if session == "pre" else "post_price")
                if ep:
                    px, label = float(ep), ("سعر البري" if session == "pre" else "سعر الأفتر")
            # كل الجلسات بنفس التأكيد: شمعة 15 دقيقة خضراء داخل المنطقة / شمعتين فوق الاختراق / إغلاق تحت الوقف (بري/أفتر)
            seg_now = session if session in ("pre", "regular", "post") else "regular"
            b = brk_bars.get(t)
            hold_ok = bool(faisal_watch.breakout_held(b, w.get("breakout"), now_et, C.INTRADAY_HOLD_BARS, seg_now))
            zok = faisal_watch.zone_confirmed(b, w.get("demand_low"), w.get("demand_high"), now_et, seg_now)
            sok = None if seg_now == "regular" else faisal_watch.stop_broken(b, w.get("stop"), now_et, seg_now)
            rok = faisal_watch.retest_confirmed(b, w.get("breakout"), now_et, seg_now)
            weak = faisal_watch.entry_quality(b, w, px, now_et, seg_now) if (zok or rok) else None
            ev = faisal_watch.evaluate(w, px, last_close, label, hold_ok, zone_ok=zok, stop_ok=sok,
                                       alerts_on=session in ("pre", "regular", "post"), retest_ok=rok, weak=weak)
            ev["session"] = session
            sig = by_t.get(t)
            if sig is None and df is not None and len(df) >= 60:
                try:
                    sig = analyze_pivot(t, df, frac, session == "regular" and df.index[-1].date() == today)
                except Exception:  # noqa
                    sig = None
            ev["scanner"] = ({"status": sig.status, "setup_ar": sig.setup_ar, "waiting": sig.waiting_for[:1],
                              "rsi": sig.rsi_now, "rsi_min": sig.rsi_min} if sig else None)
            faisal.append(ev)

    result = {
        "generated_at_utc": now_utc.isoformat(),
        "generated_at_et": now_et.isoformat(),
        "session": session,
        "provider": provider_name,
        "funnel": funnel,
        "counts": {"universe": int(len(universe)), "analyzed": analyzed, "ready": len(ready), "watch": len(watch),
                   "momentum": len(momw), "ext_quotes": len(quotes), "ext_movers_5pct": ext_movers},
        "ready": [dump(x) for x in ready],
        "watch": [dump(x) for x in watch[:40]],
        "momentum": [dump(x) for x in momw[:20]],
        "errors": errors,
        "ext_debug": ext_debug,
        "faisal": faisal,
    }
    # السوق مقفل: لا تنبيهات (البيانات قد تكون ناقصة والدخول غير ممكن)
    # التنبيهات: الفحص الكامل ينبه على أسهم الارتكاز فقط؛ الزخم ومستويات فيصل ينبه عليها الفحص المباشر كل دقيقة
    # (ALERT_KINDS=all يرجع كل الأنواع هنا — مثلاً إذا الفحص المباشر موقوف)
    kinds_all = os.getenv("ALERT_KINDS", "pivot") == "all"
    to_alert = result if kinds_all else {"ready": [s for s in result["ready"] if s.get("kind") != "momentum"], "faisal": []}
    result["new_alerts"] = update_alert_state(to_alert, now_et) if session != "closed" else []
    if not demo:
        try:
            probe = nasdaq_extended(["AAPL", "SNDL"], "pre",
                                    raw_dump=os.path.join(C.RESULTS_DIR, "debug", "nasdaq_ext_probe.json"))
            result["ext_probe"] = probe
        except Exception as e:  # noqa
            result["ext_probe"] = {"error": str(e)[:100]}
    # حفظ الشموع اليومية للأسهم الجاهزة وأقرب أسهم المراقبة للمراجعة اليدوية
    try:
        bdir = os.path.join(C.RESULTS_DIR, "bars")
        os.makedirs(bdir, exist_ok=True)
        for old_f in os.listdir(bdir):
            os.remove(os.path.join(bdir, old_f))
        for x in [s["ticker"] for s in result["ready"]] + [s["ticker"] for s in result["watch"][:10]]:
            if x in daily and daily[x] is not None:
                daily[x].tail(130).to_csv(os.path.join(bdir, f"{x}.csv"))
    except Exception as e:  # noqa
        log.warning("bars export failed: %s", e)
    return result


def update_alert_state(result, now_et, suffix=""):
    """suffix: الفحص المباشر (زخم + مستويات فيصل) له ملف حالة منفصل "_live" عن فحص الارتكاز الكامل،
    لأنهم يشتغلون على أجهزة مختلفة ويرفعون للمستودع بنفس الوقت."""
    d = os.path.join(C.RESULTS_DIR, "alerts")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{now_et.date().isoformat()}{suffix}.json")
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except Exception:  # noqa
        state = {}
    new = []
    for s in result["ready"]:
        key = f"{s['ticker']}:{s['setup']}"
        seg = (s.get("meta") or {}).get("segment")
        if s.get("kind") == "momentum":
            # الملف: «سهم صيني عالي التذبذب، وهذا وحده يجعله خطيراً» → زخم صيني يظهر بالقوائم بدون تنبيه دخول
            if any("صيني" in f for f in (s.get("flags") or [])):
                continue
            # تنبيه زخم واحد لكل سهم في الجلسة: إذا ضرب الوقف «بطل السبب»، وإذا فاتت الحركة «اترك السهم»
            key = f"{s['ticker']}:MOM"
        if seg in ("pre", "post"):  # تنبيه البري/الأفتر ما يمنع تنبيه الجلسة الرسمية
            key += f":{seg}"
        if s.get("kind") == "momentum" and any(k.startswith(f"{s['ticker']}:MOM") and k.count(":") == key.count(":")
                                               for k in state):
            continue
        if s.get("kind") != "momentum":
            # إشارة ص86: الجلسة الرسمية (شمعة 4 ساعات) أو بعد الإغلاق (شمعة يومية مكتملة → طلبات لجلسة بكرة)، مرة وحدة
            # لكل شمعة اكتمال (ما تتكرر بكرة على نفس الشمعة)
            if result.get("session") not in (None, "regular", "post"):
                continue
            p86 = (s.get("meta") or {}).get("p86") or {}
            key = f"{s['ticker']}:P86:{p86.get('bar_date', '')}"
            if _seen_before(key, now_et):
                continue
        if key not in state:
            state[key] = {"first_seen_et": now_et.isoformat(), "price": s["price"], "stop": s["stop"],
                          "t1": s.get("t1"), "setup_ar": s["setup_ar"]}
            new.append(s["ticker"])
    for f_ in result.get("faisal", []):
        if f_.get("alert"):
            key = f"FAISAL:{f_['ticker']}:{f_['state']}"
            if f_.get("session") in ("pre", "post"):
                key += f":{f_['session']}"
            if key not in state:
                state[key] = {"first_seen_et": now_et.isoformat(), "price": f_["price"], "state_ar": f_["state_ar"]}
                new.append(f"{f_['ticker']} ({f_.get('label', 'مستوى فيصل')})")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    return new


def _seen_before(key, now_et, days=4):
    d = os.path.join(C.RESULTS_DIR, "alerts")
    for k in range(1, days + 1):
        day = (now_et - timedelta(days=k)).date().isoformat()
        try:
            with open(os.path.join(d, f"{day}.json"), encoding="utf-8") as f:
                if key in json.load(f):
                    return True
        except Exception:  # noqa
            continue
    return False


def send_telegram_alerts(result):
    from . import notify
    from zoneinfo import ZoneInfo
    if not notify.enabled() or not result.get("new_alerts"):
        return
    when = datetime.fromisoformat(result["generated_at_et"]).astimezone(ZoneInfo("Asia/Riyadh")).strftime("%H:%M الرياض")
    for s in result.get("ready", []):
        if s["ticker"] in result["new_alerts"]:
            notify.send(notify.alert_text(s, when), title=notify.alert_title(s))
    for f in result.get("faisal", []):
        lab = f.get("label", "مستوى فيصل")
        if f"{f['ticker']} ({lab})" in result["new_alerts"]:
            seg = {"pre": "بري ماركت", "post": "أفتر ماركت"}.get(f.get("session"))
            title, body = notify.watch_alert(f, when, seg)
            notify.send(body, title=title)


def write_outputs(result):
    send_telegram_alerts(result)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    md = build_markdown(result)
    with open(os.path.join(C.RESULTS_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(C.RESULTS_DIR, "latest.md"), "w", encoding="utf-8") as f:
        f.write(md)
    hist = os.path.join(C.RESULTS_DIR, "history")
    os.makedirs(hist, exist_ok=True)
    day = result["generated_at_et"][:10]
    with open(os.path.join(hist, f"{day}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": result["generated_at_et"], "ready": [s["ticker"] for s in result["ready"]],
                            "new": result["new_alerts"], "watch": [s["ticker"] for s in result["watch"][:15]]},
                           ensure_ascii=False) + "\n")
    summ = os.getenv("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    return md


def main(argv=None):
    ap = argparse.ArgumentParser(description="Faisal-method US stock scanner")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--loop", type=int, default=0, help="repeat every N minutes")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    while True:
        t0 = _time.time()
        try:
            res = run_once(force=a.force, demo=a.demo, limit=a.limit)
            if res:
                md = write_outputs(res)
                print(md)
                log.info("Done in %.0fs — ready=%d watch=%d momentum=%d new=%s", _time.time() - t0,
                         res["counts"]["ready"], res["counts"]["watch"], res["counts"]["momentum"], res["new_alerts"])
        except Exception as e:  # noqa
            log.exception("scan failed: %s", e)
            if not a.loop:
                sys.exit(1)
        if not a.loop:
            break
        _time.sleep(max(60, a.loop * 60 - (_time.time() - t0)))


if __name__ == "__main__":
    main()
