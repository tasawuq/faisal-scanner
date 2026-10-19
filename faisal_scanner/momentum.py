"""
تحليل سهم الزخم (الفصول 2، 3، 17، 19) في البري ماركت والجلسة والأفتر ماركت.
"سهم الزخم يتحرك اليوم بسرعة بسبب خبر، فوليوم، بري ماركت أو أفتر ماركت."
"الدخول: أول موجة Pullback على فواب، أو اختراق وثبات."
"في الزخم لا تسمح للخطأ أن يتجاوز 1% إلى 2%."
"بري ماركت فقط: يطير قبل السوق ثم ينهار عند الافتتاح" (سلوك خطر).
GDHG: "بداية الشمعة الصاعدة قد تصبح دعماً… الأفضل انتظار الرجوع لها".
"""
from datetime import time
import numpy as np
import pandas as pd

from . import config as C
from .indicators import vwap
from .levels import resistance_levels, swing_highs
from .pivot import Signal, READY, WATCH

SEG_AR = {"pre": "بري ماركت", "regular": "الجلسة", "post": "أفتر ماركت"}
MOM_AR = {
    "MOM_VWAP_PULLBACK": "أول Pullback على الفواب",
    "MOM_BREAKOUT_HOLD": "اختراق القمة وثبات",
    "MOM_ABOVE_VWAP": "فوق الفواب — بانتظار Pullback",
    "MOM_EXTENDED": "ممتد بعيداً عن الفواب — مطاردة خطرة",
    "MOM_BELOW_VWAP": "تحت الفواب — لا دخول",
}
WINDOWS = {"pre": (time(4, 0), time(9, 30)), "regular": (time(9, 30), time(16, 0)), "post": (time(16, 0), time(20, 0))}


def _seg(df, today, seg):
    a, b = WINDOWS[seg]
    return df[[ts.date() == today and a <= ts.time() < b for ts in df.index]]


def analyze_momentum(ticker: str, intraday: pd.DataFrame, daily: pd.DataFrame, now_et, session: str,
                     session_frac: float, segment: str = None, ext_volume: float = None):
    segment = segment or ("regular" if session == "regular" else session)
    if segment not in WINDOWS or intraday is None or intraday.empty or daily is None or len(daily) < 21:
        return None
    today = now_et.date()
    bars = _seg(intraday, today, segment)
    if len(bars) < 2:
        return None
    hist = daily[[ts.date() < today for ts in daily.index]]
    if len(hist) < 20:
        return None
    avg_vol = float(hist["volume"].iloc[-20:].mean())
    if segment == "post":
        reg = _seg(intraday, today, "regular")
        if len(reg):
            prev_close = float(reg["close"].iloc[-1])
        elif daily.index[-1].date() == today:
            prev_close = float(daily["close"].iloc[-1])
        else:
            return None
    else:
        prev_close = float(hist["close"].iloc[-1])

    # Yahoo لا يعطي فوليوم البري/الأفتر (صفر) → الفواب يصبح متوسط السعر، والفوليوم غير مؤكد
    vol_known = segment == "regular" or float(bars["volume"].sum()) > 0
    if not vol_known:
        if len(bars) < 4:  # صفقة أو صفقتين فقط = ضجيج
            return None
        bars = bars.assign(volume=1.0)
    vw = vwap(bars)
    price = float(bars["close"].iloc[-1])
    v_now = float(vw.iloc[-1])
    chg = price / prev_close - 1
    gap = float(bars["open"].iloc[0]) / prev_close - 1 if segment == "regular" else 0.0
    hod = float(bars["high"].max())
    base = float(bars["open"].iloc[0])                     # بداية الشمعة/الحركة (درس GDHG)
    if segment == "regular":
        seg_vol = float(intraday[[ts.date() == today and ts.time() < time(16, 0) for ts in intraday.index]]["volume"].sum())
        frac = max(session_frac, 0.1)
    else:
        seg_vol = float(bars["volume"].sum()) if vol_known else 0.0
        if not vol_known and ext_volume and ext_volume > 0:  # فوليوم البري/الأفتر من ناسداك
            seg_vol, vol_known = float(ext_volume), True
        frac = 1.0
    rvol = seg_vol / max(avg_vol, 1)
    rvol_pace = seg_vol / max(avg_vol * frac, 1)
    min_vol = C.MOM_MIN_VOLUME if segment == "regular" else C.MOM_EXT_MIN_VOLUME
    if max(chg, gap) < C.MOM_MIN_CHANGE or (vol_known and seg_vol < min_vol):
        return None
    if not (C.MIN_PRICE <= price <= C.MAX_PRICE * 1.5):
        return None

    dist = price / v_now - 1
    pull = 1 - price / hod
    closes, vws = bars["close"].values, vw.values
    last2_above = bool(np.all(closes[-2:] > vws[-2:]))
    earlier = bars.iloc[:-3]
    first_high = float(earlier["high"].max()) if len(earlier) else hod
    breakout_hold = len(bars) > 4 and bool(np.all(closes[-3:] > first_high)) and price <= first_high * 1.02

    if price < v_now:
        code, stop = "MOM_BELOW_VWAP", None
    elif breakout_hold:
        code, stop = "MOM_BREAKOUT_HOLD", first_high * 0.99
    elif dist <= C.MOM_VWAP_ZONE and pull >= C.MOM_MIN_PULLBACK and last2_above:
        code, stop = "MOM_VWAP_PULLBACK", min(v_now, float(bars["low"].iloc[-3:].min())) * 0.995
    elif dist > C.MOM_EXTENDED:
        code, stop = "MOM_EXTENDED", v_now * 0.995
    else:
        code, stop = "MOM_ABOVE_VWAP", v_now * 0.995

    seg_ar = SEG_AR[segment]
    s = Signal(ticker=ticker, kind="momentum", setup=code, setup_ar=f"زخم {seg_ar} — {MOM_AR[code]}", price=price)
    s.trigger_level = first_high if code == "MOM_BREAKOUT_HOLD" else v_now
    s.stop = stop
    s.risk_pct = None if stop is None else round((price - stop) / price, 4)
    res = resistance_levels(daily, max(price, hod), C.LOOKBACK_LEVELS, C.SWING_K, {})
    # أقرب مقاومة يومية حتى لو كانت قريبة جداً (قمم سابقة + أعلى سعر 120 يوم) — "لا دخول تحت مقاومة مباشرة"
    hh = hist["high"].values[-C.LOOKBACK_LEVELS:]
    near = [float(hh[i]) for i in swing_highs(hh, k=C.SWING_K, edge=True) if hh[i] > price * 1.001]
    if len(hh) and float(hh.max()) > price * 1.001:
        near.append(float(hh.max()))
    # قمم آخر 3 أيام (قمة أمس = أول مقاومة لسهم انطلق أمس) — swing_highs ما يشوف آخر شمعة
    near += [float(x) for x in hh[-3:] if x > price * 1.001]
    r_near = min(near) if near else None
    tgt = [(hod, f"قمة {seg_ar}")] if hod >= price * 1.03 else []
    if r_near is not None and (not tgt or r_near < tgt[0][0] * 0.99):
        tgt = [(r_near, "مقاومة يومية قريبة")] + tgt
    tgt += [r for r in res if r[0] > (tgt[-1][0] * 1.03 if tgt else 0)]
    if not tgt and stop is not None:  # سماء مفتوحة: أهداف حسب المخاطرة
        rk = price - stop
        tgt = [(price + 2 * rk, "2× المخاطرة (لا مقاومة قريبة)"), (price + 3 * rk, "3× المخاطرة")]
    if tgt:
        s.t1, s.t1_label = float(tgt[0][0]), tgt[0][1]
    if len(tgt) > 1:
        s.t2, s.t2_label = float(tgt[1][0]), tgt[1][1]
    if stop is not None and s.t1:
        s.rr = round((s.t1 - price) / max(price - stop, 1e-9), 2)
    s.entry_low = s.trigger_level
    s.entry_high = s.trigger_level * (1 + C.MOM_VWAP_ZONE)
    s.support_low = base
    s.meta = {"segment": segment, "change": round(chg, 4), "gap": round(gap, 4), "rvol": round(rvol, 2),
              "rvol_pace": round(rvol_pace, 2), "vwap": round(v_now, 4), "hod": hod, "move_base": base,
              "segment_volume": seg_vol, "pullback_from_hod": round(pull, 4), "dist_vwap": round(dist, 4)}
    if segment == "regular":
        s.volume_note = f"RVOL {rvol_pace:.1f}x"
    elif vol_known:
        s.volume_note = f"فوليوم {seg_ar} {seg_vol/1e6:.2f}M ({rvol:.0%} من يوم عادي)"
    else:
        s.volume_note = f"فوليوم {seg_ar} غير مؤكد ({len(bars)} شمعة)"
    s.meta["volume_known"] = vol_known

    wait = []
    if code == "MOM_BELOW_VWAP":
        wait.append(f"استرداد الفواب {v_now:.2f} والثبات فوقه")
    elif code == "MOM_EXTENDED":
        wait.append(f"بعيد {dist*100:.0f}% عن الفواب — لا مطاردة، انتظر الرجوع قرب {v_now:.2f} أو بداية الحركة {base:.2f}")
    elif code == "MOM_ABOVE_VWAP":
        wait.append(f"أول Pullback قرب الفواب {v_now:.2f} مع ثبات")
    if segment == "regular" and rvol_pace < C.MOM_MIN_RVOL:
        wait.append(f"فوليوم نسبي أعلى (الحالي {rvol_pace:.1f}x)")
    if segment != "regular" and not vol_known:
        wait.append(f"تأكيد فوليوم {seg_ar} (غير متوفر من Yahoo — مفاتيح Alpaca تعطيه)")
    elif segment != "regular" and rvol < C.MOM_EXT_MIN_RVOL:
        wait.append(f"فوليوم {seg_ar} أعلى (الحالي {rvol:.0%} من يوم عادي)")
    if code in ("MOM_VWAP_PULLBACK", "MOM_BREAKOUT_HOLD") and s.rr is not None and s.rr < 1.0:
        wait.append(f"مقاومة قريبة عند {s.t1:.2f} ({s.t1_label}) — دخول تحت مقاومة مباشرة، انتظر اختراقها والثبات")
    if s.risk_pct is not None and s.risk_pct > C.MOM_MAX_RISK and code in ("MOM_VWAP_PULLBACK", "MOM_BREAKOUT_HOLD"):
        wait.append(f"الوقف {s.risk_pct*100:.1f}% أكبر من 2% — انتظر سعراً أقرب للمستوى")
    if segment == "pre":
        s.flags.append("حركة بري ماركت — قد تنهار عند الافتتاح، لا تشترِ القمة")
    elif segment == "post":
        s.flags.append("حركة أفتر ماركت — قد تتلاشى، السيولة ضعيفة")
    s.waiting_for = wait
    s.status = READY if (not wait and code in ("MOM_VWAP_PULLBACK", "MOM_BREAKOUT_HOLD")) else WATCH
    s.score = round(min(rvol_pace / 5, 2) + min(max(chg, 0) * 5, 2) + (2 if s.status == READY else 0), 2)
    s.rank = s.score
    return s
