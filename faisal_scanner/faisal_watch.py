"""
📌 قائمة فيصل الخاصة: الأسهم والمستويات اللي ينشرها فيصل في حسابه (@kisar_) أو قناته.
تُضاف يدوياً في faisal_watchlist.json (Claude يضيفها من التغريدات اللي ترسلها).
الماسح يراقبها كل ساعة ويطلق تنبيهاً عندما يصل السعر لمنطقة فيصل أو يخترق مستواه ويثبت.

مثال عنصر:
{"ticker": "GWAV", "added": "2026-10-15", "source": "تغريدة فيصل",
 "note": "ينتظر سحب سيولة إلى 2.70–3.15", "demand_low": 2.70, "demand_high": 3.15,
 "stop": 2.55, "breakout": null, "targets": [3.6, 4.2]}
"""
import json
import os
from datetime import time, timedelta

FILE = os.getenv("FAISAL_WATCHLIST", "faisal_watchlist.json")


def load(path: str = FILE) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa
        return []
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    today = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    out = []
    for s in data.get("stocks", []):
        if s.get("not_before") and today < s["not_before"]:
            continue  # مثل SMX: الملف يطلب مرور وقت كافٍ بعد شمعة الضخ/التصريف
        if s.get("ticker") and s.get("active", True):
            s["ticker"] = s["ticker"].upper().strip()
            out.append(s)
    return out


def alert_label(entry: dict) -> str:
    """النص اللي يظهر في قائمة التنبيهات: (مستوى فيصل) لأسهم فيصل، (متابعة) لغيرها."""
    if entry.get("kind") == "weekly":
        return "قائمة الارتكاز"
    return "مستوى فيصل" if "فيصل" in (entry.get("source") or "") else "متابعة"


SEG = {"pre": (time(4, 0), time(9, 30)), "regular": (time(9, 30), time(16, 0)), "post": (time(16, 0), time(20, 0))}


def _done(bars15, now_et, seg="regular"):
    """شموع 15 دقيقة المكتملة لليوم داخل الجلسة المطلوبة (بري/رسمية/أفتر)، بشرط تكون حديثة."""
    if bars15 is None or not len(bars15):
        return None
    a, b = SEG.get(seg, SEG["regular"])
    today = now_et.date()
    done = bars15[[ts.date() == today and a <= ts.time() < b and ts + timedelta(minutes=15) <= now_et
                   for ts in bars15.index]]
    # البري/الأفتر فيها فراغات (ما فيه صفقات) → آخر شمعة لازم تكون خلال آخر 20 دقيقة
    if len(done) and done.index[-1] + timedelta(minutes=35) < now_et:
        return done.iloc[0:0]
    return done


def zone_confirmed(bars15, lo, hi, now_et, seg="regular") -> bool:
    """تأكيد الدخول من منطقة الطلب (فيصل: ثبات + شمعة إيجابية): آخر شمعة 15 دقيقة مكتملة
    خضراء، لمست المنطقة، وأغلقت داخلها (أو فوقها بـ 2% كحد أقصى)."""
    if not lo or not hi:
        return False
    done = _done(bars15, now_et, seg)
    if done is None or not len(done):
        return False
    b = done.iloc[-1]
    return bool(b["close"] > b["open"] and b["low"] <= hi * 1.005 and lo <= b["close"] <= hi * 1.02)


def breakout_held(bars15, brk, now_et, n=2, seg="regular") -> bool:
    done = _done(bars15, now_et, seg)
    if not brk or done is None or len(done) < n:
        return None
    return bool((done["close"].values[-n:] > brk).all())


def stop_broken(bars15, stop, now_et, seg="regular") -> bool:
    """كسر مؤكد للوقف: آخر شمعة 15 دقيقة مكتملة أغلقت تحته."""
    done = _done(bars15, now_et, seg)
    if not stop or done is None or not len(done):
        return False
    return bool(done["close"].values[-1] < stop)


def retest_confirmed(bars15, brk, now_et, seg="regular") -> bool:
    """إعادة اختبار ناجحة لمستوى الاختراق (الملف ص18: «اشترِ الثبات أو إعادة الاختبار الناجحة»):
    السهم اخترق اليوم وابتعد فوق المستوى بـ 5%+، ثم رجع للمنطقة brk–brk×1.03 وقفلت شمعة 15 دقيقة خضراء فوق المستوى."""
    if not brk:
        return False
    done = _done(bars15, now_et, seg)
    if done is None or len(done) < 2:
        return False
    if float(done["high"].max()) < brk * 1.05:
        return False
    b = done.iloc[-1]
    return bool(b["close"] > b["open"] and b["low"] <= brk * 1.03 and brk <= b["close"] <= brk * 1.04)


def evaluate(entry: dict, price: float, last_close: float = None, price_label: str = "",
             hold_ok: bool = None, zone_ok: bool = None, alerts_on: bool = True, stop_ok: bool = None,
             retest_ok: bool = None) -> dict:
    """Compare the current price with Faisal's levels. hold_ok: last two 15-min bars above breakout.
    zone_ok: آخر شمعة 15 دقيقة خضراء داخل المنطقة. stop_ok=False: السعر تحت الوقف بدون إغلاق شمعة تحته
    (البري/الأفتر) → بدون تنبيه. alerts_on=False → معلومة فقط."""
    lab = "فيصل" if "فيصل" in (entry.get("source") or "") else ("القائمة" if entry.get("kind") == "weekly" else "المتابعة")
    lo, hi = entry.get("demand_low"), entry.get("demand_high")
    stop, brk = entry.get("stop"), entry.get("breakout")
    res = {"ticker": entry["ticker"], "price": price, "price_label": price_label, "note": entry.get("note", ""),
           "source": entry.get("source", ""), "added": entry.get("added", ""), "label": alert_label(entry),
           "warn": entry.get("warn", ""), "levels": {
               "demand_low": lo, "demand_high": hi, "stop": stop, "breakout": brk, "targets": entry.get("targets", [])}}
    if price is None:
        res.update(state="NO_DATA", state_ar="لا توجد بيانات سعر", alert=False)
        return res
    if entry.get("no_alert"):
        alerts_on = False  # قائمة الارتكاز: سهم بسلوك خطر (ضخ/صيني/سيولة ضعيفة) → يظهر بالقوائم بدون تنبيه
    if stop and price < stop:
        if stop_ok is False:
            res.update(state="BELOW_STOP", alert=False,
                       state_ar=f"تحت وقف {lab} {stop:g} — ينتظر إغلاق شمعة 15 دقيقة تحته للتأكيد")
        else:
            # قائمة الارتكاز: ما فيه مركز → كسر الوقف معلومة في القائمة، مو تنبيه
            res.update(state="BELOW_STOP", state_ar=f"❌ تحت وقف {lab} {stop:g} — السبب بطل",
                       alert=alerts_on and entry.get("kind") != "weekly")
    elif lo and hi and lo <= price <= hi:
        if zone_ok:
            res.update(state="ZONE_CONFIRMED", alert=alerts_on,
                       state_ar=f"✅ ثبت داخل منطقة {lab} {lo:g}–{hi:g} وأغلقت شمعة 15 دقيقة خضراء — دخول ممكن"
                                + (f"، الوقف {stop:g}" if stop else ""))
        else:
            res.update(state="IN_ZONE", alert=False,
                       state_ar=f"داخل منطقة {lab} {lo:g}–{hi:g} — ينتظر شمعة 15 دقيقة خضراء تثبت داخل المنطقة (لا دخول قبلها)")
    elif brk and retest_ok and brk <= price <= brk * 1.04:
        res.update(state="RETEST_HOLD", alert=alerts_on,
                   state_ar=f"✅ إعادة اختبار ناجحة لمستوى {lab} {brk:g} — شمعة 15 دقيقة خضراء فوقه")
    elif brk and price > brk:
        held = hold_ok if hold_ok is not None else (last_close is not None and last_close > brk)
        if held:
            # قائمة الارتكاز: لا شراء للاختراق نفسه (ص20) → التنبيه على إعادة الاختبار فقط
            res.update(state="BREAKOUT_HOLD", state_ar=f"🚀 اخترق مستوى {lab} {brk:g} وثبت",
                       alert=alerts_on and entry.get("kind") != "weekly")
        else:
            res.update(state="BREAKOUT_PENDING", state_ar=f"فوق {brk:g} — ينتظر الثبات أكثر من ربع ساعة", alert=False)
    else:
        parts = []
        if lo and price > (hi or lo):
            parts.append(f"فوق منطقة الطلب بـ {(price / (hi or lo) - 1) * 100:.0f}% — ينتظر النزول لـ {lo:g}–{(hi or lo):g}")
        elif lo and price < lo:
            parts.append(f"تحت منطقة الطلب ({lo:g})")
        if brk and price <= brk:
            parts.append(f"تحت مستوى الاختراق {brk:g} بـ {(brk / price - 1) * 100:.0f}%")
        res.update(state="WAITING", state_ar="؛ ".join(parts) or f"ينتظر مستويات {lab}", alert=False)
    return res
