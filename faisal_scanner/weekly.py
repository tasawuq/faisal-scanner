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
from . import method as M
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
    """الفرز قبل الصعود (ص56، ص59 «هكذا تفرز الاسهم قبل صعودها»، ص90–94): عدد أسهم/فلووت صغير جداً + تقسيم عكسي
    حديث + سهم سبق صعد 100%+ + قرب القاع + شروط ص86. «توه مدبل» = غير آمن الآن."""
    flt = sig.meta.get("float") or sig.meta.get("shares")
    s = 0.0
    if flt:
        s += 4.0 if flt <= 1e6 else 3.0 if flt <= 3e6 else 2.0 if flt <= 5e6 else 1.0 if flt <= 10e6 else 0.5
    tags = " ".join(sig.tags)
    flags = " ".join(sig.flags)
    if "Former Runner" in tags:
        s += 2.0
    if "تقسيم عكسي حديث" in tags:
        s += 2.0
    s += 0.5 * (sig.score or 0)  # عدد شروط ص86 المتحققة (من 7)
    if "توه مدبل" in flags:
        s -= 3.0
    if "صيني" in flags:
        s -= 1.5
    if "ضخ/تصريف" in flags:
        s -= 2.0
    if sig.price < 0.2:
        s -= 1.0
    return round(s, 2)


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
        z_lo, z_hi = sig.meta.get("zone") or (sig.entry_low, sig.entry_high)
        w_lo, w_hi, w_note = wash_zone(df, z_lo)
        lad = sig.meta.get("ladder") or {}
        p86 = sig.meta.get("p86") or {}
        met, miss = M.summary(p86.get("checks") or {})
        items.append({
            "ticker": sig.ticker, "score": sc, "price": round(float(sig.price), 4),
            "ext_price": sig.meta.get("ext_price"),
            "name": sig.meta.get("name") or "",
            "type": stock_type(sig, df), "setup_ar": sig.setup_ar, "status": sig.status,
            "data": {k: sig.meta.get(k) for k in ("float", "shares", "last_split", "split", "country_ar", "industry",
                                                   "short_available", "runner")},
            "news": (news or {}).get(sig.ticker, []),
            "levels": {"demand_low": round(float(z_lo), 4), "demand_high": round(float(z_hi), 4),
                       "orders": lad.get("orders"), "critical": lad.get("critical"), "stop": lad.get("stop"),
                       "risk": lad.get("risk"), "wash_low": round(w_lo, 4), "wash_high": round(w_hi, 4),
                       "wash_note": w_note, "pivot": sig.trigger_level, "t1": sig.t1, "t2": sig.t2},
            "p86": {"frame": p86.get("frame"), "bar_date": p86.get("bar_date"), "met": met, "miss": miss,
                    "checks": p86.get("checks")},
            "not_ready": sig.waiting_for[:4], "ready_when": sig.meta.get("when_ready", ""),
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
    from .main import data_line, news_line
    lines = [f"# قائمة الارتكاز (أسلوب فيصل) — {w['generated_at_et'][:16]} نيويورك", "",
             "كل الأسهم هنا **متابعة فقط — غير جاهزة للدخول** (ص54، ص70). الإشارة الوحيدة: اكتمال شروط ص86 على اليومي "
             "أو 4 ساعات، وبعدها طلبات Limit مقسّمة داخل المنطقة.", ""]
    for it in w["items"]:
        lv, p = it["levels"], it.get("p86") or {}
        lines.append(f"## ${it['ticker']} — {' / '.join(it['type'])}")
        lines.append(f"- السعر {_p(it['price'])}" + (f" (بري/أفتر {_p(it['ext_price'])})" if it.get("ext_price") else "")
                     + f" | {it['setup_ar']}")
        lines.append("- البيانات: " + data_line(it.get("data") or {}))
        nl = news_line(it.get("news"))
        if nl:
            lines.append("- الأخبار: " + nl)
        o = lv.get("orders") or []
        lines.append(f"- منطقة الطلب {_p(lv['demand_low'])}–{_p(lv['demand_high'])} | طلبات Limit: "
                     + " / ".join(_p(x) for x in o) + f" | الخروج: إغلاق يومي تحت {_p(lv.get('critical'))}"
                     + f" | الأهداف {_p(lv.get('t1'))} / {_p(lv.get('t2'))}"
                     + f" | منطقة أعمق (غسل) {_p(lv['wash_low'])}–{_p(lv['wash_high'])}")
        lines.append(f"- شروط ص86 ({p.get('frame') or 'يومي'}): {len(p.get('met') or [])}/7"
                     + (" — ناقص: " + "، ".join(M.NAME[k] for k in p.get("miss") or []) if p.get("miss") else ""))
        if it["not_ready"]:
            lines.append("- لماذا غير جاهز: " + "؛ ".join(it["not_ready"]))
        lines.append("- متى يكون جاهزاً: " + it["ready_when"])
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
        # 27 أكتوبر: أسهم القائمة متابعة فقط (ص54) — ما فيه تنبيه دخول من القائمة؛ الإشارة الوحيدة من الفحص (شروط ص86)
        out.append({"ticker": it["ticker"], "kind": "weekly", "source": "قائمة الارتكاز (تطبيق الملف)", "no_alert": True,
                    "demand_low": lv["demand_low"], "demand_high": lv["demand_high"], "stop": lv.get("stop"),
                    "breakout": None, "targets": sorted({round(float(x), 4) for x in (lv.get("t1"), lv.get("t2")) if x}),
                    "note": it.get("ready_when", ""), "warn": "، ".join(it.get("flags") or []),
                    "last_price": it.get("ext_price") or it["price"], "wash_low": lv.get("wash_low"),
                    "data": it.get("data") or {}})
    return out
