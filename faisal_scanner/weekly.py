"""
قائمة الارتكاز بأسلوب فيصل (ص70–73، ص90–94، تغريدة الفرز ص59، تغريدة «غير جاهز للدخول» ص54):
- كل سهم: نوع السلوك، البيانات (عدد الأسهم/الفلووت، الشورت، آخر تقسيم، السلوك السابق، الأخبار)،
  المستويات (القاع، منطقة الطلب للـ Limit، منطقة الغسل/سحب السيولة تحتها، المستوى الفاصل، المقاومات)،
  لماذا غير جاهز، ومتى يصير جاهز.
- الفرز قبل الصعود: عدد أسهم/فلووت صغير، شورت قليل، تقسيم عكسي حديث، سلوك سابق (انفجر قبل)، قريب من القاع.
- «أي سهم ارتكاز يتم طرحه غير جاهز للدخول، فقط متابعة… انتظر أقل سعر 10–20% تحت» → الدخول بأوامر Limit من تحت.
"""
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np

from . import config as C
from .levels import swing_lows

log = logging.getLogger("faisal")
FILE = os.path.join(C.RESULTS_DIR, "weekly.json")
MAX_ITEMS = None  # كل الأسهم اللي تطابق (طلب المستخدم: «تفحص الأسهم كلها وليس 10 وكل ساعة»)


def _num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except Exception:  # noqa
        return None


def wash_zone(df, z_low):
    """منطقة الغسل (سحب السيولة تحت الدعم — ص17، ص55، أمثلة GWAV 2.70 / MWC 2.35 / MBRX 2.34 / CDTG 1.40):
    أقرب قاع تاريخي تحت الدعم ضمن 25%، وإلا 10–15% تحت الدعم (تغريدة ص54: «انتظر أقل سعر 10% أو 20% تحت»)."""
    lows = df["low"].values.astype(float)
    n = len(lows)
    st = max(0, n - 250)
    cands = [lows[i] for i in swing_lows(lows, k=3) if st <= i < n - 3 and z_low * 0.75 <= lows[i] < z_low * 0.97]
    if cands:
        w = max(cands)
        return w * 0.99, min(w * 1.02, z_low * 0.985), "قاع سابق تحت الدعم"
    return z_low * 0.85, z_low * 0.90, "10–15% تحت الدعم (ص54)"


def stock_type(sig, df):
    flags = " ".join(sig.flags)
    flt = sig.meta.get("float")
    types = []
    if "صيني" in flags:
        types.append("صيني عالي التذبذب ⚠️")
    if "طبي" in flags:
        types.append("طبي — الخبر يحرّكه")
    if len(df) < 150:
        types.append("طرح جديد/تاريخ قصير")
    if (sig.meta.get("gain_from_low10") or 0) >= 0.30:
        types.append("بعد ارتداد أول — ينتظر Retest")
    if flt and flt < C.HOT_FLOAT:
        types.append("فلووت منخفض جداً")
    if "قاع الفترة" in " ".join(sig.tags):
        types.append("قاع نظيف")
    if sig.setup.startswith("BASE") or sig.setup.startswith("W_"):
        types.append("🚩 علم/اختراق بعد قاعدة" if sig.setup.startswith("BASE") else "نموذج W")
    return types or ["ارتكاز"]


def sort_score(sig, fund):
    """الفرز قبل الصعود (ص59 + ص9): فلووت/عدد أسهم صغير + سلوك سابق + تقسيم + شورت قليل + قرب القاع + الشروط."""
    f = fund or {}
    flt = sig.meta.get("float") or _num(f.get("sharesOutstanding"))
    s = 0.0
    if flt:
        s += 3.0 if flt < 3e6 else 2.0 if flt < 5e6 else 1.5 if flt < 10e6 else 1.0 if flt < 20e6 else 0.0
    else:
        s += 0.5
    tags = " ".join(sig.tags)
    flags = " ".join(sig.flags)
    if "Former Runner" in tags:
        s += 1.5
    if "تقسيم عكسي حديث" in flags:
        s += 1.0
    sp = sig.meta.get("short_pct_float")
    if sp is not None and sp < 0.03:
        s += 0.5
    g = sig.meta.get("gain_from_low10") or 0
    s += 1.0 if g <= 0.15 else (-1.0 if g > 0.30 else 0.0)
    if (sig.meta.get("drawdown") or 0) >= 0.40:
        s += 1.0
    if (sig.rsi_min or 99) <= 30:
        s += 1.0
    s += 0.5 * (sig.score or 0)
    if "صيني" in flags:
        s -= 1.5
    if "ضخ/تصريف حديثة" in flags:
        s -= 2.0
    if "سيولة ضعيفة" in flags:
        s -= 1.0
    if sig.price < 0.2:
        s -= 1.0
    return round(s, 2)


def ready_when(sig, wash):
    z_lo, z_hi = sig.entry_low, sig.entry_high
    st = sig.setup
    w_lo, w_hi = wash
    if st in ("BASE_BREAK", "BASE_RETEST", "W_BREAK", "W_RETEST"):
        return (f"🚩 رجوع لاختبار {sig.trigger_level:.3g} والثبات فوقه (شمعة 15 دقيقة خضراء) — لا شراء للاختراق نفسه (ص20)")
    if st == "W_FORMING":
        return (f"من تحت: شمعة ارتداد عند القاع الثاني {sig.support_low:.3g}؛ أو كسر خط العنق {sig.trigger_level:.3g} "
                f"والثبات أكثر من ربع ساعة ثم إعادة اختبار (ص25)")
    if st == "BROKE_SUPPORT":
        return f"استرداد {z_lo:.3g} والإغلاق فوقه؛ وإذا استمر النزول فالطلب الثاني عند {w_lo:.3g}–{w_hi:.3g}"
    return (f"نزول لمنطقة الطلب {z_lo:.3g}–{z_hi:.3g} ثم شمعة ارتداد/ثبات، أو سحب سيولة إلى {w_lo:.3g}–{w_hi:.3g} "
            f"ثم استرداد {z_lo:.3g} (ص17، ص86)")


_NEWS = {}


def nasdaq_news(symbols, limit=3):
    """عناوين الأخبار من ناسداك (بدون مفتاح). يرجع {sym: [(date, title)]}، وفاضي إذا فشل. كاش ساعة."""
    import requests
    from .data_sources import NASDAQ_HDR
    out = {}
    hour = datetime.now(timezone.utc).strftime("%Y%m%d")  # كاش يومي للأخبار
    for s in symbols:
        if (s, hour) in _NEWS:
            out[s] = _NEWS[(s, hour)]
            continue
        try:
            r = requests.get("https://api.nasdaq.com/api/news/topic/articlebysymbol",
                             params={"q": f"{s}|stocks", "offset": 0, "limit": limit, "fallback": "false"},
                             headers=NASDAQ_HDR, timeout=15)
            rows = (((r.json() or {}).get("data") or {}).get("rows") or [])[:limit]
            out[s] = [(x.get("created") or x.get("ago") or "", (x.get("title") or "").strip()) for x in rows if x.get("title")]
            _NEWS[(s, hour)] = out[s]
        except Exception as e:  # noqa
            log.debug("nasdaq news %s: %s", s, e)
    return out


def due(now_et, max_age_hours=1):
    """تُبنى من جديد كل ساعة (طلب المستخدم) أو مع يوم جديد."""
    try:
        with open(FILE, encoding="utf-8") as fh:
            t = datetime.fromisoformat(json.load(fh)["generated_at_et"])
    except Exception:  # noqa
        return True
    return t.date() != now_et.date() or (now_et - t).total_seconds() > max_age_hours * 3600


def build(pivots, daily, fund, now_et, news=None):
    rows = []
    for sig in pivots:
        if sig.kind != "pivot" or sig.entry_low is None or sig.stop is None:
            continue
        df = daily.get(sig.ticker)
        if df is None or len(df) < 60:
            continue
        rows.append((sort_score(sig, (fund or {}).get(sig.ticker)), sig))
    rows.sort(key=lambda r: -r[0])
    rows = rows[:MAX_ITEMS] if MAX_ITEMS else rows
    if news is None:
        try:
            news = nasdaq_news([s.ticker for _, s in rows])
        except Exception:  # noqa
            news = {}
    items = []
    for sc, sig in rows:
        df = daily[sig.ticker]
        f = (fund or {}).get(sig.ticker) or {}
        d_lo, d_hi, stop_t, pivot = float(sig.entry_low), float(sig.entry_high), float(sig.stop), sig.t1
        if sig.setup == "W_FORMING" and sig.support_low:
            # نموذج W قيد التكوين: الدخول من تحت عند القاع الثاني (منطقة الطلب)، وخط العنق هو المستوى الفاصل (ص25)
            d_lo, d_hi = float(sig.support_low), float(max(sig.support_high, sig.support_low * 1.02))
            stop_t, pivot = d_lo * (1 - C.STOP_BUFFER), sig.trigger_level
        z_lo = sig.support_low if sig.setup in ("AT_SUPPORT", "SWEEP", "BOUNCE", "BROKE_SUPPORT", "W_FORMING") else d_lo
        w_lo, w_hi, w_note = wash_zone(df, z_lo)
        base_low = float(np.min(df["low"].values[-60:]))
        # مقاومة مباشرة فوق منطقة الدخول (ص67/ص88: لا دخول تحت مقاومة مباشرة): أقرب قمة يومية من آخر 5 أيام
        # فوق أعلى المنطقة — مثال TPET 21 أكتوبر: الدخول 1.44–1.49 وقمم الأيام السابقة 1.54–1.60 تحت خط العنق 1.67
        near_hi = [float(x) for x in df["high"].values[-6:-1] if x > d_hi * 1.005]
        r1 = round(min(near_hi), 4) if near_hi else None
        tags = " ".join(sig.tags)
        runner = next((t for t in sig.tags if "Former Runner" in t), "")
        split = next((x for x in sig.flags if "تقسيم عكسي" in x), "")
        items.append({
            "ticker": sig.ticker, "score": sc, "price": round(float(sig.price), 4),
            "ext_price": sig.meta.get("ext_price"),
            "name": sig.meta.get("name") or f.get("longName") or "",
            "type": stock_type(sig, df), "setup_ar": sig.setup_ar, "status": sig.status,
            "data": {"float": sig.meta.get("float"), "shares": _num(f.get("sharesOutstanding")),
                     "short_pct_float": sig.meta.get("short_pct_float"), "split": split, "runner": runner,
                     "drawdown": sig.meta.get("drawdown"), "rsi_min": sig.rsi_min, "rsi_now": sig.rsi_now,
                     "ma": sig.ma_state, "macd": sig.macd_state},
            "news": (news or {}).get(sig.ticker, []),
            "levels": {"bottom": round(base_low, 4), "demand_low": round(d_lo, 4),
                       "demand_high": round(d_hi, 4), "wash_low": round(w_lo, 4),
                       "wash_high": round(w_hi, 4), "wash_note": w_note, "pivot": pivot, "t1": sig.t1, "t2": sig.t2, "r1": r1,
                       "stop": round(min(stop_t, w_lo * 0.97), 4), "stop_tight": round(stop_t, 4)},
            "not_ready": sig.waiting_for[:2], "ready_when": ready_when(sig, (w_lo, w_hi)),
            "flags": sig.flags[:4], "tags": sig.tags[:4],
        })
    out = {"generated_at_et": now_et.isoformat(), "items": items}
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(FILE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(C.RESULTS_DIR, "weekly.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown(out))
    return out


def _p(x):
    if x is None:
        return "—"
    x = float(x)
    return f"{x:.4f}".rstrip("0").rstrip(".") if x < 1 else f"{x:.3f}".rstrip("0").rstrip(".")


def markdown(w):
    lines = [f"# قائمة الارتكاز (أسلوب فيصل) — {w['generated_at_et'][:16]} نيويورك", "",
             "كل الأسهم هنا **غير جاهزة للدخول الآن — متابعة فقط** (تغريدة فيصل ص54). الدخول بأوامر Limit من تحت عند الشرط.", ""]
    for it in w["items"]:
        d, lv = it["data"], it["levels"]
        flt = d.get("float") or d.get("shares")
        lines.append(f"## ${it['ticker']} — {' / '.join(it['type'])}")
        lines.append(f"- السعر {_p(it['price'])}" + (f" (بري/أفتر {_p(it['ext_price'])})" if it.get("ext_price") else "")
                     + f" | {it['setup_ar']}")
        lines.append(f"- البيانات: فلووت {flt / 1e6:.2f}M" if flt else "- البيانات: فلووت غير معروف")
        extra = []
        if d.get("short_pct_float") is not None:
            extra.append(f"شورت {d['short_pct_float'] * 100:.1f}% من الفلووت")
        if d.get("split"):
            extra.append(d["split"])
        if d.get("runner"):
            extra.append(d["runner"])
        extra.append(f"هبط {(d.get('drawdown') or 0) * 100:.0f}% من قمة 90 يوم، أدنى RSI {d.get('rsi_min')}")
        lines[-1] += " — " + " — ".join(extra)
        if it["news"]:
            lines.append("- الأخبار: " + " | ".join(f"{t} ({dt})" for dt, t in it["news"][:2]))
        lines.append(f"- القاع {_p(lv['bottom'])} | منطقة الطلب (Limit) {_p(lv['demand_low'])}–{_p(lv['demand_high'])}"
                     f" | منطقة الغسل {_p(lv['wash_low'])}–{_p(lv['wash_high'])} ({lv['wash_note']})"
                     f" | المستوى الفاصل {_p(lv['pivot'])} | الأهداف {_p(lv['t1'])} / {_p(lv['t2'])}"
                     f" | الوقف: تحت {_p(lv['stop_tight'])} لدخول منطقة الطلب، وتحت {_p(lv['stop'])} لدخول الغسل")
        if it["not_ready"]:
            lines.append("- غير جاهز لأن: " + "؛ ".join(it["not_ready"]))
        lines.append("- يصير جاهز: " + it["ready_when"])
        if it["flags"]:
            lines.append("- ⚠️ " + "، ".join(it["flags"]))
        lines.append("")
    return "\n".join(lines)


def load_entries(path=FILE, max_age_days=4):
    """مداخل قائمة الارتكاز للمتابعة المباشرة (تنبيه عند الوصول لمنطقة الطلب + شمعة تأكيد)."""
    try:
        with open(path, encoding="utf-8") as fh:
            w = json.load(fh)
        t = datetime.fromisoformat(w["generated_at_et"])
        if (datetime.now(timezone.utc) - t.astimezone(timezone.utc)).days > max_age_days:
            return []
    except Exception:  # noqa
        return []
    out = []
    for it in w.get("items", []):
        lv = it["levels"]
        fl = " ".join(it.get("flags") or [])
        # سلوك خطر حسب الملف (ص10، ص70، ص63): ضخ/تصريف خلال الشهر، صيني، سيولة ضعيفة، أسعار متقطعة → مراقبة بدون تنبيه
        no_alert = any(k in fl for k in ("ضخ/تصريف", "صيني", "سيولة ضعيفة", "متقطعة"))
        out.append({"ticker": it["ticker"], "kind": "weekly", "source": "قائمة الارتكاز (تطبيق الملف)", "no_alert": no_alert,
                    "demand_low": lv["demand_low"], "demand_high": lv["demand_high"], "stop": lv["stop_tight"],
                    "breakout": lv.get("pivot"), "targets": sorted({x for x in (lv.get("r1"), lv.get("t1"), lv.get("t2")) if x}),
                    "note": it["ready_when"], "warn": "، ".join(it.get("flags") or []),
                    "last_price": it.get("ext_price") or it["price"], "wash_low": lv["wash_low"]})
    return out
