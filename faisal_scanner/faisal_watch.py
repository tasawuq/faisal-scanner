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


MIN_CONFIRM_DOLLARS = 20_000   # شمعة التأكيد لازم فيها سيولة حقيقية (قيمة تداول)
MIN_CONFIRM_VOL_X = 1.5        # فوليوم شمعة التأكيد ≥ 1.5× وسيط فوليوم شموع 15 دقيقة (الجلسة الرسمية)
MIN_RR = 1.5                   # قائمة الارتكاز: المسافة لأول مقاومة/هدف ≥ 1.5× المسافة للوقف


def entry_quality(bars15, entry: dict, price: float, now_et, seg: str = "regular") -> list:
    """فلتر جودة تنبيه الدخول (منطقة الطلب / إعادة الاختبار) — بعد مراجعة تنبيهات 21 أكتوبر:
    1) الجلسة الرسمية فقط (ص10: الحركة في البري/الأفتر فقط سلوك خطر) — OLB/PRSO/SHMD نبهت في البري وفشلت.
    2) شمعة التأكيد بفوليوم حقيقي (ص19: بدون فوليوم = ضعيف) — RENT/VNTG/TPET نبهت بشموع 1–8 آلاف سهم.
    3) قائمة الارتكاز: العائد لأول مقاومة ≥ 1.5× المخاطرة للوقف — SURG هدفه +3% ووقفه −5.5%.
    ترجع قائمة الأسباب (فاضية = التنبيه سليم)."""
    why = []
    if seg != "regular":
        why.append("خارج الجلسة الرسمية (ص10)")
    done = _done(bars15, now_et, seg) if bars15 is not None else None
    if done is not None and len(done):
        b = done.iloc[-1]
        vol = float(b["volume"] or 0)
        reg = bars15[[SEG["regular"][0] <= ts.time() < SEG["regular"][1] for ts in bars15.index]]
        reg = reg[reg.index < done.index[-1]]
        med = float(reg["volume"].median()) if len(reg) >= 6 else 0.0
        if vol * float(b["close"]) < MIN_CONFIRM_DOLLARS:
            why.append(f"شمعة التأكيد بدون سيولة ({vol / 1e3:.1f}K سهم ≈ ${vol * float(b['close']) / 1e3:.0f}K)")
        elif med and vol < MIN_CONFIRM_VOL_X * med:
            why.append(f"فوليوم شمعة التأكيد ضعيف ({vol / 1e3:.0f}K مقابل المعتاد {med / 1e3:.0f}K — ص19)")
    # 27 أكتوبر («فيصل يقول بعد التقسيم... والعدد الأسهم»): ص6 فلتر الارتكاز Float Under 20M (50M فقط إذا ما طلعت نتائج)،
    # ص8 «أقل من 20M مقبول للمضاربة» → سهم القائمة بفلووت أكبر من 20M مراقبة فقط بدون تنبيه (NERV 34M نبّه 10-27)
    fl = (entry.get("data") or {}).get("float")
    if entry.get("kind") == "weekly" and fl and fl > 20e6:
        why.append(f"الفلووت {fl / 1e6:.1f}M أكبر من 20M (ص6/ص8)")
    # 26 أكتوبر (UPXI نبّه وهو 3/6): سهم القائمة ما ينبّه إلا إذا اجتمعت أغلب شروط الملف الستة (4 من 6، ص81)
    if entry.get("kind") == "weekly" and entry.get("file_met") is not None and entry["file_met"] < 4:
        why.append(f"شروط الملف {entry['file_met']}/6 — ناقص: {'، '.join(entry.get('file_miss') or [])} (ص81)")
    if entry.get("kind") == "weekly" and price and entry.get("stop") and price > entry["stop"]:
        obst = [x for x in [entry.get("breakout")] + list(entry.get("targets") or []) if x and x > price * 1.005]
        if obst:
            reward, risk = min(obst) - price, price - entry["stop"]
            if reward < MIN_RR * risk:
                why.append(f"المساحة لأول مقاومة {min(obst):g} (+{reward / price * 100:.1f}%) أقل من 1.5× المخاطرة "
                           f"(−{risk / price * 100:.1f}%)")
    return why


def confirm_info(bars15, entry: dict, price: float, now_et, seg: str = "regular") -> dict:
    """27 أكتوبر («ابي نفس التنبيهات ذي صحيحة بدون اخطاء» بعد شرح GANX): تفاصيل «ليش طلع التنبيه» داخل نصه —
    شمعة التأكيد (وقتها، افتتاح→إغلاق، فوليومها وقيمتها ونسبتها للمعتاد) والمساحة لأول مقاومة مقابل الوقف."""
    out = {}
    try:
        done = _done(bars15, now_et, seg) if bars15 is not None else None
        if done is not None and len(done):
            b = done.iloc[-1]
            vol, c = float(b["volume"] or 0), float(b["close"])
            reg = bars15[[SEG["regular"][0] <= ts.time() < SEG["regular"][1] for ts in bars15.index]]
            reg = reg[reg.index < done.index[-1]]
            med = float(reg["volume"].median()) if len(reg) >= 6 else 0.0
            out["candle"] = {"time": done.index[-1].strftime("%H:%M"), "open": float(b["open"]), "close": c,
                             "vol": vol, "dollars": vol * c, "x_med": (vol / med) if med else None}
        stop = entry.get("stop")
        if price and stop and price > stop:
            obst = [x for x in [entry.get("breakout")] + list(entry.get("targets") or []) if x and x > price * 1.005]
            if obst:
                r = min(obst)
                out["space"] = {"t1": r, "up": (r / price - 1), "down": (1 - stop / price),
                                "rr": (r - price) / (price - stop)}
    except Exception:  # noqa
        pass
    if entry.get("file_met") is not None:
        out["file"] = {"met": entry["file_met"], "miss": entry.get("file_miss") or []}
    return out


def scanner_confirm(bars15, price: float, stop: float, t1: float, entry_low: float, daily=None, now_et=None) -> list:
    """بوابة الجودة لتنبيهات «جاهز» من الماسح (23 أكتوبر، طلب المستخدم «نعم وموكده»):
    نفس بوابة القائمة — كانت تنبيهات الماسح تكفيها الشروط اليومية (ELTX نبّه 1.98 وكسر الوقف، TYGO نبّه 0.986
    والمساحة لهدفه 1.09 تساوي المخاطرة تقريباً). الجاهز فعلياً لازم:
    1) شمعة 15 دقيقة خضراء مكتملة (من آخر شمعتين) أغلقت فوق أسفل منطقة الطلب، فوليومها ≥1.5× المعتاد و≥ $20K،
       وما تراجع السعر بعدها تحت افتتاحها.
    2) المسافة من السعر الحالي لأول مقاومة (الهدف الأول أو أقرب قمة يومية من آخر 5 أيام) ≥ 1.5× المسافة للوقف.
    ترجع الأسباب (فاضية = جاهز ومؤكد)."""
    why = []
    done = _done(bars15, now_et, "regular") if bars15 is not None and now_et is not None else None
    if done is None or not len(done):
        why.append("تأكيد 15 دقيقة: ما فيه شمعة مكتملة اليوم بعد (بوابة الجودة)")
    else:
        reg_all = bars15[[SEG["regular"][0] <= ts.time() < SEG["regular"][1] for ts in bars15.index]]
        ok, reasons = False, []  # السبب المعروض = سبب آخر شمعة مكتملة
        for i in (len(done) - 1, len(done) - 2):
            if i < 0:
                continue
            last_reason = ""
            b = done.iloc[i]
            o, c, v = float(b["open"]), float(b["close"]), float(b["volume"] or 0)
            prev = reg_all[reg_all.index < done.index[i]]
            med = float(prev["volume"].median()) if len(prev) >= 6 else 0.0
            if c <= o:
                last_reason = "آخر شمعة 15 دقيقة مو خضراء"
            elif entry_low and c < entry_low:
                last_reason = f"شمعة التأكيد أغلقت تحت المنطقة {entry_low:.3g}"
            elif v * c < MIN_CONFIRM_DOLLARS:
                last_reason = f"شمعة التأكيد بدون سيولة (≈ ${v * c / 1e3:.0f}K)"
            elif med and v < MIN_CONFIRM_VOL_X * med:
                last_reason = f"فوليوم شمعة التأكيد ضعيف ({v / 1e3:.0f}K مقابل المعتاد {med / 1e3:.0f}K — ص19)"
            elif i < len(done) - 1 and float(done["close"].iloc[-1]) < o:
                last_reason = "السعر رجع تحت افتتاح شمعة التأكيد"
            else:
                ok = True
                break
            reasons.append(last_reason)
        if not ok:
            why.append(f"تأكيد 15 دقيقة: شمعة خضراء بفوليوم ≥1.5× المعتاد و≥ $20K — {reasons[0] if reasons else 'غير متحقق'}")
    if price and stop:
        if price <= stop:
            why.append(f"السعر تحت الوقف {stop:.3g}")
        else:
            obst = [t1] if t1 else []
            if daily is not None and len(daily):
                d = daily
                if now_et is not None and d.index[-1].date() == now_et.date():
                    d = d.iloc[:-1]
                obst += [float(h) for h in d["high"].values[-5:]]
            obst = [x for x in obst if x and x > price * 1.005]
            if obst:
                r = min(obst)
                reward, risk = r - price, price - stop
                if reward < MIN_RR * risk:
                    pmax = (r + MIN_RR * stop) / (1 + MIN_RR)
                    why.append(f"المساحة لأول مقاومة {r:.3g} (+{reward / price * 100:.1f}%) أقل من 1.5× المخاطرة "
                               f"(−{risk / price * 100:.1f}%) — الدخول عند {pmax:.3g} أو أقل")
    return why


def evaluate(entry: dict, price: float, last_close: float = None, price_label: str = "",
             hold_ok: bool = None, zone_ok: bool = None, alerts_on: bool = True, stop_ok: bool = None,
             retest_ok: bool = None, weak: list = None) -> dict:
    """Compare the current price with Faisal's levels. hold_ok: last two 15-min bars above breakout.
    zone_ok: آخر شمعة 15 دقيقة خضراء داخل المنطقة. stop_ok=False: السعر تحت الوقف بدون إغلاق شمعة تحته
    (البري/الأفتر) → بدون تنبيه. alerts_on=False → معلومة فقط."""
    lab = "فيصل" if "فيصل" in (entry.get("source") or "") else ("القائمة" if entry.get("kind") == "weekly" else "المتابعة")
    lo, hi = entry.get("demand_low"), entry.get("demand_high")
    stop, brk = entry.get("stop"), entry.get("breakout")
    res = {"ticker": entry["ticker"], "price": price, "price_label": price_label, "note": entry.get("note", ""),
           "source": entry.get("source", ""), "added": entry.get("added", ""), "label": alert_label(entry),
           "warn": entry.get("warn", ""), "data": entry.get("data") or {}, "levels": {
               "demand_low": lo, "demand_high": hi, "stop": stop, "breakout": brk, "targets": entry.get("targets", [])}}
    if price is None:
        res.update(state="NO_DATA", state_ar="لا توجد بيانات سعر", alert=False)
        return res
    if entry.get("no_alert"):
        alerts_on = False  # قائمة الارتكاز: متابعة فقط (ص54) — الإشارة من الفحص عند اكتمال شروط ص86
    if entry.get("kind") == "weekly":
        res["alert"] = False
        if stop and price < stop:
            res.update(state="BELOW_STOP", state_ar=f"تحت وقف القائمة {stop:g} — الدعم مكسور، الانتظار أعمق (ص54)")
        elif lo and hi and lo <= price <= hi:
            res.update(state="IN_ZONE", state_ar=f"داخل منطقة الطلب {lo:g}–{hi:g} — متابعة فقط؛ الإشارة عند اكتمال "
                                                 f"شروط ص86 بشمعة يومية أو 4 ساعات")
        elif lo and price > hi:
            res.update(state="WAITING", state_ar=f"فوق منطقة الطلب بـ {(price / hi - 1) * 100:.0f}% — الطلبات تنتظر "
                                                 f"{lo:g}–{hi:g} (لا مطاردة)")
        else:
            res.update(state="WAITING", state_ar=f"تحت منطقة الطلب ({lo:g}) — ينتظر استرداد أو المنطقة الأعمق")
        return res
    claude = "فيصل" not in (entry.get("source") or "")
    if stop and price < stop:
        if stop_ok is False:
            res.update(state="BELOW_STOP", alert=False,
                       state_ar=f"تحت وقف {lab} {stop:g} — ينتظر إغلاق شمعة 15 دقيقة تحته للتأكيد")
        else:
            # قائمة الارتكاز: ما فيه مركز → كسر الوقف معلومة في القائمة، مو تنبيه
            res.update(state="BELOW_STOP", state_ar=f"❌ تحت وقف {lab} {stop:g} — السبب بطل",
                       alert=alerts_on and entry.get("kind") != "weekly" and entry.get("alert_stop", True))
    elif lo and hi and lo <= price <= hi:
        if zone_ok and weak:
            res.update(state="ZONE_WEAK", alert=False,
                       state_ar=f"⚪ شمعة خضراء داخل منطقة {lab} {lo:g}–{hi:g} لكن التأكيد ضعيف — بدون تنبيه: "
                                + "؛ ".join(weak))
        elif zone_ok:
            res.update(state="ZONE_CONFIRMED", alert=alerts_on,
                       state_ar=f"✅ ثبت داخل منطقة {lab} {lo:g}–{hi:g} وأغلقت شمعة 15 دقيقة خضراء بفوليوم — دخول ممكن"
                                + (f"، الوقف {stop:g}" if stop else ""))
        else:
            res.update(state="IN_ZONE", alert=False,
                       state_ar=f"داخل منطقة {lab} {lo:g}–{hi:g} — ينتظر شمعة 15 دقيقة خضراء تثبت داخل المنطقة (لا دخول قبلها)")
    elif brk and retest_ok and weak and brk <= price <= brk * 1.04:
        res.update(state="RETEST_WEAK", alert=False,
                   state_ar=f"⚪ إعادة اختبار {brk:g} بشمعة خضراء لكن التأكيد ضعيف — بدون تنبيه: " + "؛ ".join(weak))
    elif brk and retest_ok and brk <= price <= brk * 1.04:
        res.update(state="RETEST_HOLD", alert=alerts_on,
                   state_ar=f"✅ إعادة اختبار ناجحة لمستوى {lab} {brk:g} — شمعة 15 دقيقة خضراء فوقه")
    elif brk and price > brk:
        held = hold_ok if hold_ok is not None else (last_close is not None and last_close > brk)
        if held:
            # قائمة الارتكاز: لا شراء للاختراق نفسه (ص20) → التنبيه على إعادة الاختبار فقط
            res.update(state="BREAKOUT_HOLD", state_ar=f"🚀 اخترق مستوى {lab} {brk:g} وثبت",
                       alert=alerts_on and entry.get("kind") != "weekly" and entry.get("alert_breakout", True))
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
    if claude and res.get("state") in ("ZONE_CONFIRMED", "IN_ZONE", "RETEST_HOLD", "BREAKOUT_HOLD", "ZONE_WEAK",
                                       "RETEST_WEAK", "BREAKOUT_PENDING"):
        # 27 أكتوبر (منهج الملف حرفياً): مستويات المتابعة من عندي ما تعطي دخول بشمعة 15 دقيقة (ص90: اقربها 4 ساعات) —
        # متابعة فقط؛ الإشارة من الفحص عند اكتمال شروط ص86. (مستويات فيصل المنشورة تبقى بقاعدته هو: ثبات >ربع ساعة ص25)
        res["alert"] = False
        res["state_ar"] = (f"{res['state_ar'].split(' — ')[0]} — متابعة فقط؛ الإشارة عند اكتمال شروط ص86 "
                           f"(يومي/4 ساعات)")
    return res
