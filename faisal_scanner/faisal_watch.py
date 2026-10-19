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
    out = []
    for s in data.get("stocks", []):
        if s.get("ticker") and s.get("active", True):
            s["ticker"] = s["ticker"].upper().strip()
            out.append(s)
    return out


def alert_label(entry: dict) -> str:
    """النص اللي يظهر في قائمة التنبيهات: (مستوى فيصل) لأسهم فيصل، (متابعة) لغيرها."""
    return "مستوى فيصل" if "فيصل" in (entry.get("source") or "") else "متابعة"


def _done_regular(bars15, now_et):
    """شموع 15 دقيقة المكتملة لليوم في الجلسة الرسمية فقط."""
    if bars15 is None or not len(bars15):
        return None
    today = now_et.date()
    return bars15[[ts.date() == today and time(9, 30) <= ts.time() < time(16, 0)
                   and ts + timedelta(minutes=15) <= now_et for ts in bars15.index]]


def zone_confirmed(bars15, lo, hi, now_et) -> bool:
    """تأكيد الدخول من منطقة الطلب (فيصل: ثبات + شمعة إيجابية): آخر شمعة 15 دقيقة مكتملة
    خضراء، لمست المنطقة، وأغلقت داخلها (أو فوقها بـ 2% كحد أقصى)."""
    if not lo or not hi:
        return False
    done = _done_regular(bars15, now_et)
    if done is None or not len(done):
        return False
    b = done.iloc[-1]
    return bool(b["close"] > b["open"] and b["low"] <= hi * 1.005 and lo <= b["close"] <= hi * 1.02)


def breakout_held(bars15, brk, now_et, n=2) -> bool:
    done = _done_regular(bars15, now_et)
    if not brk or done is None or len(done) < n:
        return None
    return bool((done["close"].values[-n:] > brk).all())


def evaluate(entry: dict, price: float, last_close: float = None, price_label: str = "",
             hold_ok: bool = None, zone_ok: bool = None, alerts_on: bool = True) -> dict:
    """Compare the current price with Faisal's levels. hold_ok: last two 15-min bars above breakout.
    zone_ok: آخر شمعة 15 دقيقة خضراء داخل المنطقة. alerts_on=False خارج الجلسة الرسمية (معلومة فقط)."""
    lab = "فيصل" if "فيصل" in (entry.get("source") or "") else "المتابعة"
    lo, hi = entry.get("demand_low"), entry.get("demand_high")
    stop, brk = entry.get("stop"), entry.get("breakout")
    res = {"ticker": entry["ticker"], "price": price, "price_label": price_label, "note": entry.get("note", ""),
           "source": entry.get("source", ""), "added": entry.get("added", ""), "label": alert_label(entry), "levels": {
               "demand_low": lo, "demand_high": hi, "stop": stop, "breakout": brk, "targets": entry.get("targets", [])}}
    if price is None:
        res.update(state="NO_DATA", state_ar="لا توجد بيانات سعر", alert=False)
        return res
    if stop and price < stop:
        res.update(state="BELOW_STOP", state_ar=f"❌ تحت وقف {lab} {stop:g} — السبب بطل", alert=alerts_on)
    elif lo and hi and lo <= price <= hi:
        if zone_ok:
            res.update(state="ZONE_CONFIRMED", alert=alerts_on,
                       state_ar=f"✅ ثبت داخل منطقة {lab} {lo:g}–{hi:g} وأغلقت شمعة 15 دقيقة خضراء — دخول ممكن"
                                + (f"، الوقف {stop:g}" if stop else ""))
        else:
            res.update(state="IN_ZONE", alert=False,
                       state_ar=f"داخل منطقة {lab} {lo:g}–{hi:g} — ينتظر شمعة 15 دقيقة خضراء تثبت داخل المنطقة (لا دخول قبلها)")
    elif brk and price > brk:
        held = hold_ok if hold_ok is not None else (last_close is not None and last_close > brk)
        if held:
            res.update(state="BREAKOUT_HOLD", state_ar=f"🚀 اخترق مستوى {lab} {brk:g} وثبت", alert=alerts_on)
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
