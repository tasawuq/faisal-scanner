"""
تحليل سهم الارتكاز حسب منهج فيصل (الفصول 7، 9، 13، 21–25، 29).

المعادلة (صفحة 66):
    ضغط قوي + دعم واضح + RSI 23–27 + تحسن MACD/فوليوم + شمعة إيجابية
    + محفز أو هدف علوي + دخول قريب من الدعم = سهم ارتكاز جيد

"لا تدخل لأن السهم تحرك؛ ادخل لأن السبب الفني اكتمل."
لذلك السهم يكون "جاهز" فقط عند اكتمال أحد نماذج الدخول الصحيح (صفحة 67):
    A) سحب سيولة تحت الدعم ثم استرداده
    B) ارتداد مؤكد من دعم حقيقي (شمعة إيجابية + ثبات)
    C) كسر خط عنق W / قمة الارتداد ثم الثبات أو إعادة الاختبار (نموذج الأسهم الهابطة ص68)
    D) استعادة MA20 ثم إعادة اختباره كدعم (ص64)
وإلا يكون "تحت المراقبة" مع ذكر ما ينتظره بالضبط.
"""
from dataclasses import dataclass, field, asdict
import math
import numpy as np
import pandas as pd

from . import config as C
from .indicators import sma, rsi, macd, atr
from .levels import (swing_lows, build_zones, count_visits, resistance_levels, candle, candle_name, Zone)

READY, WATCH = "READY", "WATCH"

SETUP_AR = {
    "SWEEP": "سحب سيولة ثم استرداد الدعم",
    "BOUNCE": "ارتداد مؤكد من الدعم",
    "W_BREAK": "كسر خط العنق (W) والثبات",
    "W_RETEST": "إعادة اختبار خط العنق بنجاح",
    "MA_RETEST": "استعادة MA20 وإعادة اختباره كدعم",
    "AT_SUPPORT": "ارتكاز عند الدعم — بانتظار التأكيد",
    "BROKE_SUPPORT": "كسر الدعم — ينتظر الاسترداد (سحب سيولة أو فشل)",
    "W_FORMING": "نموذج W قيد التكوين",
    "MA_RECLAIM": "استعاد MA20 — بانتظار إعادة الاختبار",
    "EXTENDED": "حقق صعوداً — لا مطاردة",
}


@dataclass
class Signal:
    ticker: str
    kind: str = "pivot"
    status: str = WATCH
    setup: str = ""
    setup_ar: str = ""
    price: float = float("nan")
    score: float = 0.0
    conditions: dict = field(default_factory=dict)
    support_low: float = None
    support_high: float = None
    support_touches: int = 0
    trigger_level: float = None
    stop: float = None
    entry_low: float = None
    entry_high: float = None
    t1: float = None
    t1_label: str = ""
    t2: float = None
    t2_label: str = ""
    rr: float = None
    risk_pct: float = None
    rsi_now: float = None
    rsi_min: float = None
    ma_state: str = ""
    macd_state: str = ""
    candle: str = ""
    volume_note: str = ""
    waiting_for: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    rank: float = 0.0
    meta: dict = field(default_factory=dict)
    news: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d


# ====================================================================== context
class Ctx:
    """All per-ticker measurements, computed once."""

    def __init__(self, df: pd.DataFrame, session_frac: float = 1.0, last_bar_partial: bool = False):
        self.df = df
        o, h, l, c, v = (df[x].values.astype(float) for x in ("open", "high", "low", "close", "volume"))
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.n = n = len(df)
        close = df["close"].astype(float)
        self.ma = {p: sma(close, p).values for p in C.MA_PERIODS}
        self.rsi = rsi(close, C.RSI_PERIOD).values
        line, sig, hist = macd(close, C.MACD_FAST, C.MACD_SLOW, C.MACD_SIGNAL)
        self.macd_line, self.macd_sig, self.hist = line.values, sig.values, hist.values
        self.atr = atr(df).values
        self.price = c[-1]
        frac = max(session_frac, 0.15) if last_bar_partial else 1.0
        self.v_last_adj = v[-1] / frac
        self.avg_vol = float(np.nanmean(v[-21:-1])) if n > 21 else float(np.nanmean(v))

        ma20, ma30, ma50 = (self.ma[p] for p in C.MA_PERIODS)
        mins = np.fmin(np.fmin(ma20, ma30), ma50)
        self.below_all_now = bool(c[-1] < np.nanmin([ma20[-1], ma30[-1], ma50[-1]]))
        self.above_all_now = bool(c[-1] > np.nanmax([ma20[-1], ma30[-1], ma50[-1]]))
        self.below_all_recent = bool(np.any(c[-20:] < mins[-20:]))
        self.reclaim20 = bool(c[-1] > ma20[-1] and np.any(c[-10:-1] < ma20[-10:-1]))
        self.dist_ma20 = (ma20[-1] / c[-1] - 1.0) if np.isfinite(ma20[-1]) else float("nan")
        # فشل عند المتوسط: لمس MA20 ورفض بذيل علوي
        self.fail_at_ma20 = False
        for j in (-1, -2):
            rng = max(h[j] - l[j], 1e-9)
            if h[j] >= ma20[j] * 0.99 and c[j] < ma20[j] and (h[j] - max(o[j], c[j])) / rng >= 0.4:
                self.fail_at_ma20 = True

        if self.reclaim20:
            self.ma_state = "يستعيد MA20 بعد أن كان تحت المتوسطات"
        elif self.above_all_now and self.below_all_recent:
            self.ma_state = "استعاد المتوسطات 20/30/50"
        elif self.below_all_now:
            self.ma_state = "تحت كل المتوسطات (ضغط)"
        elif c[-1] > ma20[-1]:
            self.ma_state = "فوق MA20"
        else:
            self.ma_state = "بين المتوسطات"
        if self.fail_at_ma20:
            self.ma_state += " — رفض عند MA20"

        # MACD
        hh = self.hist
        self.hist_up2 = bool(hh[-1] > hh[-2] > hh[-3])
        self.hist_up1 = bool(hh[-1] > hh[-2])
        self.macd_bull = bool(self.macd_line[-1] > self.macd_sig[-1])
        self.macd_cross = any(
            self.macd_line[j - 1] < self.macd_sig[j - 1] and self.macd_line[j] >= self.macd_sig[j]
            for j in range(n - 3, n)
        )
        if self.macd_cross:
            self.macd_state = "تقاطع إيجابي حديث"
        elif self.hist_up2:
            self.macd_state = "الهيستوجرام يتحسن"
        elif self.hist_up1:
            self.macd_state = "بداية تحسن"
        elif self.macd_bull:
            self.macd_state = "إيجابي (الخط فوق الإشارة)"
        else:
            self.macd_state = "سلبي"

        # candles of last two bars
        self.candles = []
        for j in (n - 2, n - 1):
            self.candles.append((j, candle(o[j], h[j], l[j], c[j], o[j - 1], c[j - 1])))

        # pressure
        self.drawdown = 1.0 - c[-1] / np.nanmax(h[-90:])
        self.recent_low10 = float(np.min(l[-10:]))
        self.gain_from_low10 = c[-1] / self.recent_low10 - 1.0

        # volume behaviour
        red = c < o
        r5 = v[-5:][red[-5:]]
        r15 = v[-20:-5][red[-20:-5]]
        self.sell_drying = bool(len(r5) and len(r15) and r5.mean() < 0.8 * r15.mean())
        green_last = c[-1] > o[-1] or c[-1] > c[-2]
        red_mean = float(np.mean(v[-10:][red[-10:]])) if np.any(red[-10:]) else self.avg_vol
        self.vol_improving = bool(green_last and self.v_last_adj >= max(self.avg_vol, red_mean))
        notes = []
        if self.sell_drying:
            notes.append("فوليوم البيع يضعف")
        if self.vol_improving:
            notes.append("فوليوم يتحسن مع الارتداد")
        self.volume_note = "، ".join(notes)

        # higher lows
        sl = [i for i in swing_lows(l, k=2, edge=True) if i >= n - 30]
        self.higher_lows = len(sl) >= 2 and l[sl[-1]] > l[sl[-2]]

        # danger candles (قروبات الوهم / تصريف)
        self.pump_recent = False
        self.distribution_recent = False
        for j in range(max(1, n - 5), n):
            rng = max(h[j] - l[j], 1e-9)
            up = (h[j] - max(o[j], c[j])) / rng
            vm = (self.v_last_adj if j == n - 1 else v[j]) / max(self.avg_vol, 1)
            chg = c[j] / c[j - 1] - 1
            spike = h[j] / c[j - 1] - 1
            if (chg >= C.PUMP_CHANGE or spike >= C.PUMP_CHANGE * 1.5) and vm >= C.PUMP_VOL_MULT and up >= 0.4:
                self.pump_recent = True
            if j >= n - 3 and vm >= 3 and (c[j] - l[j]) / rng <= 0.33 and spike >= 0.15:
                self.distribution_recent = True

        # Former Runner (ص4، LFS): سهم سبق انفجر بعنف خلال آخر سنة
        st = max(1, n - C.RUNNER_LOOKBACK)
        jumps = c[st:] / c[st - 1:-1] - 1
        spikes = h[st:] / c[st - 1:-1] - 1
        self.runner_jump = float(max(np.nanmax(jumps), 0)) if len(jumps) else 0.0
        self.runner_spike = float(max(np.nanmax(spikes), 0)) if len(spikes) else 0.0
        self.former_runner = self.runner_jump >= C.RUNNER_CLOSE_JUMP or self.runner_spike >= C.RUNNER_SPIKE
        # شمعة ضخ/تصريف خلال آخر شهر (قروبات الوهم) — مقارنة بالفوليوم الوسيط
        self.pump_month = None
        med = float(np.nanmedian(v[-60:-1])) if n > 61 else float(np.nanmedian(v))
        for j in range(max(1, n - C.PUMP_LOOKBACK), n):
            rng = max(h[j] - l[j], 1e-9)
            up = (h[j] - max(o[j], c[j])) / rng
            spike = h[j] / c[j - 1] - 1
            vm = (self.v_last_adj if j == n - 1 else v[j]) / max(med, 1)
            dumped = up >= 0.5 or (h[j] - c[j]) / max(h[j], 1e-9) >= 0.25  # ذيل علوي طويل أو فتح فوق ثم تصريف
            if spike >= 0.30 and vm >= C.PUMP_VOL_MULT and dumped:
                self.pump_month = (j, float(h[j]), float(c[j]), vm)

    def rsi_min(self, lookback: int) -> float:
        return float(np.nanmin(self.rsi[-lookback:]))


# ====================================================================== scoring
def score_conditions(ctx: Ctx, rsi_lb: int, zone_low: float, zone_high: float, rr, late_rsi: float,
                     short_pct=None, breakout: bool = False) -> dict:
    cond = {}
    # 1) المتوسطات 20/30/50
    if ctx.reclaim20 or (ctx.above_all_now and ctx.below_all_recent) or (ctx.c[-1] > ctx.ma[20][-1] and ctx.below_all_recent):
        cond["ma"] = (1.0, ctx.ma_state)
    elif ctx.below_all_now:
        cond["ma"] = (0.5, ctx.ma_state)
    else:
        cond["ma"] = (0.0, ctx.ma_state)
    if ctx.fail_at_ma20:
        cond["ma"] = (0.0, ctx.ma_state)

    # 2) RSI 23–27
    m = ctx.rsi_min(rsi_lb)
    now = float(ctx.rsi[-1])
    if C.RSI_IDEAL_LOW <= m <= C.RSI_IDEAL_HIGH:
        s = 1.0
    elif m <= C.RSI_READY_MAX:
        s = 0.75  # 20–23 أو 27–28 أو أعمق من 20: ضغط حقيقي لكن خارج المنطقة المثالية
    elif m <= C.RSI_OVERSOLD:
        s = 0.5   # 28–30: قريب لكن لم يصل منطقة فيصل
    elif m <= 35:
        s = 0.25
    else:
        s = 0.0
    note = f"أدنى RSI {m:.0f} والحالي {now:.0f}"
    if now > late_rsi:
        s = min(s, 0.25)
        note += " (ارتفع — جزء من الارتداد حصل)"
    cond["rsi"] = (s, note)

    # 3) MACD
    if ctx.macd_cross or ctx.hist_up2 or (breakout and ctx.macd_bull):
        cond["macd"] = (1.0, ctx.macd_state)
    elif ctx.hist_up1 or ctx.macd_bull:
        cond["macd"] = (0.5, ctx.macd_state)
    else:
        cond["macd"] = (0.0, ctx.macd_state)

    # 4) شمعة إيجابية عند الدعم
    best, name = 0.0, ""
    for j, info in ctx.candles:
        at_zone = ctx.l[j] <= zone_high * 1.03 and ctx.c[j] >= zone_low * 0.99
        a = ctx.atr[j] if np.isfinite(ctx.atr[j]) else 0.0
        clear = (ctx.h[j] - ctx.l[j]) >= C.REVERSAL_MIN_ATR * a
        if info["bullish_reversal"] and at_zone and clear:
            if best < 1.0:
                best, name = 1.0, candle_name(info)
        elif (info["bullish_reversal"] or info["strong_green"]) and at_zone and best < 0.5:
            best, name = 0.5, candle_name(info) + (" (شمعة صغيرة غير واضحة)" if not clear else "")
    cond["candle"] = (best, name or "لا توجد شمعة إيجابية عند الدعم")

    # 5) ضغط بيع / شورت
    sp = short_pct or 0.0
    if ctx.drawdown >= C.DRAWDOWN_STRONG or sp >= C.SHORT_STRONG:
        s = 1.0
    elif ctx.drawdown >= C.DRAWDOWN_MED or sp >= C.SHORT_MED:
        s = 0.5
    else:
        s = 0.0
    note = f"هبط {ctx.drawdown*100:.0f}% من قمة 90 يوم"
    if sp:
        note += f"، شورت {sp*100:.0f}% من الفلووت"
    cond["pressure"] = (s, note)

    # 6) مساحة صعود / هدف علوي (الخبر يُضاف لاحقاً)
    if rr is None:
        cond["space"] = (0.0, "لا يوجد هدف واضح فوق")
    elif rr >= 3:
        cond["space"] = (1.0, f"العائد/المخاطرة {rr:.1f}")
    elif rr >= 2:
        cond["space"] = (0.75, f"العائد/المخاطرة {rr:.1f}")
    elif rr >= C.MIN_RR_T1:
        cond["space"] = (0.5, f"العائد/المخاطرة {rr:.1f}")
    else:
        cond["space"] = (0.0, f"العائد/المخاطرة {rr:.1f} — قريب من مقاومة")
    return cond


def total(cond: dict) -> float:
    return round(sum(v[0] for v in cond.values()), 2)


# ====================================================================== support
def find_support(ctx: Ctx):
    n = ctx.n
    start = max(0, n - C.LOOKBACK_LEVELS)
    hist_end = n - 3
    pts = [(i, ctx.l[i]) for i in swing_lows(ctx.l[:n], k=C.SWING_K) if start <= i < hist_end]
    if not pts:
        return None
    zones = build_zones(pts, C.ZONE_TOL)
    period_min = float(np.min(ctx.l[start:hist_end]))
    for z in zones:
        z.touches = max(z.touches, count_visits(ctx.l, ctx.c, z, start, hist_end, C.ZONE_TOL))
        z.is_bottom = z.low <= period_min * 1.001
    ref = float(np.min(ctx.l[-3:]))
    near = [z for z in zones
            if ref <= z.high * (1 + C.NEAR_SUPPORT) and ref >= z.low * (1 - C.MAX_SWEEP_DEPTH)
            and (z.touches >= 2 or z.is_bottom)]
    if not near:
        # أقرب دعم تحت السعر (للمراقبة فقط)
        below = [z for z in zones if z.high <= ctx.price and (z.touches >= 2 or z.is_bottom)
                 and ctx.price <= z.high * 1.15]
        if not below:
            return None
        z = max(below, key=lambda z: z.high)
        return z, False
    z = min(near, key=lambda z: (-(min(z.touches, 4)), abs(ref / z.mid - 1)))
    return z, True


# ====================================================================== W pattern
def detect_w(ctx: Ctx):
    n = ctx.n
    start = max(0, n - 60)
    sl = [i for i in swing_lows(ctx.l, k=C.SWING_K, edge=True) if i >= start]
    if len(sl) < 2:
        return None
    wmin = float(np.min(ctx.l[start:]))
    for b in reversed(sl):
        for a in reversed(sl):
            if a >= b - 4:
                continue
            pa, pb = ctx.l[a], ctx.l[b]
            if min(pa, pb) > wmin * 1.03:
                continue
            if pb < pa * (1 - C.W_MAX_LOW_DIFF) or pb > pa * 1.15:
                continue
            seg = ctx.h[a:b + 1]
            ni = a + int(np.argmax(seg))
            neck = float(ctx.h[ni])
            if neck < max(pa, pb) * (1 + C.W_MIN_NECK_HEIGHT):
                continue
            if b + 1 < n and np.min(ctx.l[b + 1:]) < pb * 0.97:
                continue
            return {"a": a, "b": b, "neck": neck, "neck_idx": ni, "low1": float(pa), "low2": float(pb)}
    return None


# ====================================================================== main entry
def analyze_pivot(ticker: str, df: pd.DataFrame, session_frac: float = 1.0, last_bar_partial: bool = False,
                  short_pct=None):
    if df is None or len(df) < 60:
        return None
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < 60:
        return None
    ctx = Ctx(df, session_frac, last_bar_partial)
    price = ctx.price
    ma_vals = {"MA20": ctx.ma[20][-1], "MA30": ctx.ma[30][-1], "MA50": ctx.ma[50][-1]}
    cands = []

    # ---------------- A/B: الدعم وسحب السيولة
    sup = find_support(ctx)
    if sup is not None:
        z, interacting = sup
        recent_low = float(np.min(ctx.l[-3:]))
        broken = ctx.c[-1] < z.low * 0.99 and ctx.c[-2] < z.low * 0.99
        swept = recent_low < z.low * 0.995 and ctx.c[-1] >= z.low
        reversal = any(info["bullish_reversal"] and ctx.l[j] <= z.high * 1.03 for j, info in ctx.candles)
        # ثبات: الإغلاق فوق الدعم وليس عند قاع الشمعة
        holding = ctx.c[-1] >= z.low and (ctx.c[-1] - ctx.l[-1]) >= 0.4 * (ctx.h[-1] - ctx.l[-1])
        stop = min(z.low, recent_low) * (1 - C.STOP_BUFFER)
        entry_low = z.low
        entry_high = min(price, z.high * 1.03)
        if entry_high < entry_low:
            entry_high = entry_low * 1.02
        if broken:
            pass  # كسر الدعم وبقي تحته = فشل، لا يُدرج
        elif interacting and swept:
            cands.append(dict(code="SWEEP", zone=z, trigger=z.low, stop=stop, entry=(entry_low, entry_high),
                              confirmed=holding, rsi_lb=C.RSI_LOOKBACK_BOUNCE, late=C.RSI_LATE, breakout=False))
        elif interacting and reversal and holding:
            cands.append(dict(code="BOUNCE", zone=z, trigger=z.low, stop=stop, entry=(entry_low, entry_high),
                              confirmed=True, rsi_lb=C.RSI_LOOKBACK_BOUNCE, late=C.RSI_LATE, breakout=False))
        elif price < z.low:
            # كسر الدعم بشمعة واحدة: إما سحب سيولة (إذا استرد) أو فشل (إذا بقي تحته)
            cands.append(dict(code="BROKE_SUPPORT", zone=z, trigger=z.low, stop=recent_low * (1 - C.STOP_BUFFER),
                              entry=(z.low, z.high), confirmed=False, rsi_lb=C.RSI_LOOKBACK_BOUNCE,
                              late=C.RSI_LATE, breakout=False))
        else:
            cands.append(dict(code="AT_SUPPORT", zone=z, trigger=z.low, stop=z.low * (1 - C.STOP_BUFFER),
                              entry=(z.low, z.high), confirmed=False, rsi_lb=C.RSI_LOOKBACK_BOUNCE,
                              late=C.RSI_LATE, breakout=False, near=interacting))

    # ---------------- C: نموذج W / كسر قمة الارتداد
    w = detect_w(ctx)
    if w is not None:
        neck, b = w["neck"], w["b"]
        after = range(b + 1, ctx.n)
        brk = next((j for j in after if ctx.c[j] > neck), None)
        wz = Zone(min(w["low1"], w["low2"]), max(w["low1"], w["low2"]), 2, w["a"], b)
        if brk is None:
            if price >= w["low2"] * 1.02 and price >= neck * 0.85:
                cands.append(dict(code="W_FORMING", zone=wz, trigger=neck, stop=neck * 0.97,
                                  entry=(neck, neck * 1.03), confirmed=False, rsi_lb=C.RSI_LOOKBACK_BREAKOUT,
                                  late=C.RSI_LATE_BREAKOUT, breakout=True, neck=neck))
        else:
            closes_after = ctx.c[brk:]
            held = bool(np.all(closes_after >= neck * 0.98)) and ctx.c[-1] >= neck
            retest = brk + 1 < ctx.n and np.min(ctx.l[brk + 1:]) <= neck * 1.03 and held
            ext = price / neck - 1
            low_since = float(np.min(ctx.l[brk:]))
            stop = min(neck * 0.97, low_since * 0.98)
            nz = Zone(neck * 0.98, neck * 1.01, 2, brk, ctx.n - 1)
            brk_vol_ok = bool(ctx.v[brk] >= 1.2 * max(ctx.avg_vol, 1) or (brk == ctx.n - 1 and ctx.v_last_adj >= 1.2 * ctx.avg_vol))
            if held:
                code = "W_RETEST" if retest else "W_BREAK"
                cands.append(dict(code=code, zone=nz, trigger=neck, stop=stop, entry=(neck, min(price, neck * 1.03)),
                                  confirmed=ext <= C.MAX_EXTENSION_BREAKOUT, rsi_lb=C.RSI_LOOKBACK_BREAKOUT,
                                  late=C.RSI_LATE_BREAKOUT, breakout=True, neck=neck, ext=ext,
                                  vol_ok=brk_vol_ok))
            # إن لم يثبت = اختراق وهمي → يُعلَّم كتحذير
            elif ctx.c[-1] < neck and brk >= ctx.n - 10:
                ctx.bull_trap = True

    # (المتوسطات عند فيصل فلتر تأكيد وليست دخولاً مستقلاً — ص64، لذلك لا يوجد نموذج دخول خاص بها)

    if not cands:
        return None

    results = [_evaluate(ticker, ctx, cd, ma_vals, short_pct) for cd in cands]
    results = [r for r in results if r is not None]
    if not results:
        return None
    results.sort(key=lambda r: (r.status == READY, r.rank), reverse=True)
    best = results[0]
    if getattr(ctx, "bull_trap", False):
        best.flags.append("اختراق وهمي سابق لخط العنق (Bull Trap)")
    return best


def _evaluate(ticker, ctx: Ctx, cd: dict, ma_vals: dict, short_pct):
    price = ctx.price
    z: Zone = cd["zone"]
    stop = cd["stop"]
    entry_low, entry_high = cd["entry"]
    # "ضع أوامر Limit داخل منطقة الطلب ولا تشتري من العرض" → المخاطرة تُحسب من منتصف منطقة الطلب
    entry_ref = (entry_low + entry_high) / 2
    if entry_ref <= stop:
        return None
    res = resistance_levels(ctx.df, max(price, entry_high), C.LOOKBACK_LEVELS, C.SWING_K, ma_vals)
    # مقاومة مباشرة يفوّتها تحليل القمم: قمم آخر 5 أيام (قبل شمعة الإشارة) فوق السعر ولو كانت قريبة جداً
    # (مثال SVRE/VIP: قمة أمس فوق السعر بـ 2–4% = أول هدف حقيقي، والدخول تحتها "تحت مقاومة مباشرة")
    ref_px = max(price, entry_high) * 1.005
    # (في نماذج الاختراق W قمة ما بعد الكسر هي الهدف الطبيعي، وحماية المطاردة موجودة أصلاً)
    near = [] if cd.get("breakout") else [float(x) for x in ctx.h[-6:-1] if x > ref_px]
    if near:
        nr = min(near)
        if not res or nr < res[0][0] * 0.99:
            res = [(nr, "قمة يوم قريبة")] + list(res)
    t1 = res[0] if res else None
    t2 = res[1] if len(res) > 1 else None
    risk = (entry_ref - stop) / entry_ref
    rr = ((t1[0] - entry_ref) / (entry_ref - stop)) if t1 else None
    rrs = [(r[0] - entry_ref) / (entry_ref - stop) for r in res[:3]]
    rr_best = max(rrs) if rrs else None
    cond = score_conditions(ctx, cd["rsi_lb"], z.low, z.high, rr_best, cd["late"], short_pct, cd["breakout"])
    score = total(cond)

    s = Signal(ticker=ticker, setup=cd["code"], setup_ar=SETUP_AR[cd["code"]], price=float(price))
    s.conditions = {k: {"score": v[0], "note": v[1]} for k, v in cond.items()}
    s.score = score
    s.support_low, s.support_high, s.support_touches = float(z.low), float(z.high), int(z.touches)
    s.trigger_level = float(cd["trigger"])
    s.stop = float(stop)
    s.entry_low, s.entry_high = float(entry_low), float(entry_high)
    # أعلى منطقة الدخول ما يتعدى السعر اللي يكون فيه الربح لأول مقاومة ≥ المخاطرة (وإلا صار «دخول تحت مقاومة مباشرة»)
    # مثال VIP 2026-10-19: الدعم 1.22، الوقف 1.1956، أول مقاومة 1.32 → أعلى دخول 1.258 (مو 1.2875)
    if t1 and not cd.get("breakout"):
        cap = (t1[0] + C.MIN_RR_T1 * stop) / (1 + C.MIN_RR_T1)
        if s.entry_high > cap:
            s.entry_high = float(max(entry_low, cap))
    if t1:
        s.t1, s.t1_label = float(t1[0]), t1[1]
    if t2:
        s.t2, s.t2_label = float(t2[0]), t2[1]
    s.rr = None if rr is None else round(float(rr), 2)
    s.risk_pct = round(float(risk), 4)
    s.rsi_now = round(float(ctx.rsi[-1]), 1)
    s.rsi_min = round(ctx.rsi_min(cd["rsi_lb"]), 1)
    s.ma_state, s.macd_state = ctx.ma_state, ctx.macd_state
    s.candle = cond["candle"][1] if cond["candle"][0] > 0 else ""
    s.volume_note = ctx.volume_note
    s.meta = {"avg_volume": ctx.avg_vol, "drawdown": round(float(ctx.drawdown), 3),
              "rr_best": None if rr_best is None else round(float(rr_best), 2),
              "gain_from_low10": round(float(ctx.gain_from_low10), 3), "higher_lows": ctx.higher_lows}
    if ctx.former_runner:
        s.tags.insert(0, f"Former Runner (سبق انفجر +{max(ctx.runner_jump, ctx.runner_spike) * 100:.0f}% في يوم)")
    if ctx.higher_lows:
        s.tags.append("قيعان أعلى")
    if ctx.sell_drying:
        s.tags.append("فوليوم البيع يضعف")
    if ctx.vol_improving:
        s.tags.append("فوليوم يتحسن")
    if z.is_bottom and not cd["breakout"]:
        s.tags.append("قاع الفترة (قاع تاريخي/سنوي)")

    # -------------------------------------------- gates (شروط الجاهزية)
    wait = []
    if not cd["confirmed"]:
        if cd["code"] == "AT_SUPPORT":
            if cd.get("near"):
                wait.append(f"شمعة ارتداد (Hammer/Engulfing) أو سحب سيولة تحت {z.low:.2f} ثم استرداد")
            else:
                wait.append(f"نزول السعر لمنطقة الطلب {z.low:.2f}–{z.high:.2f}")
        elif cd["code"] == "BROKE_SUPPORT":
            wait.append(f"استرداد الدعم {z.low:.2f} والإغلاق فوقه — إذا بقي تحته فهو فشل وليس سحب سيولة")
        elif cd["code"] == "W_FORMING":
            wait.append(f"كسر خط العنق {cd['neck']:.2f} والثبات فوقه أكثر من ربع ساعة")
        elif cd["code"] == "MA_RECLAIM":
            wait.append(f"إعادة اختبار MA20 ({cd['trigger']:.2f}) والارتداد منه")
        elif cd["code"] in ("W_BREAK", "W_RETEST", "MA_RETEST"):
            wait.append(f"ابتعد {cd.get('ext', 0)*100:.0f}% عن مستوى الاختراق — انتظر إعادة اختبار {cd['trigger']:.2f}")
        elif cd["code"] == "SWEEP":
            wait.append(f"إغلاق وثبات فوق {z.low:.2f} بعد سحب السيولة")
    if cond["rsi"][0] < 0.75:
        wait.append(f"ضغط RSI في منطقة 23–27 (أدنى قراءة حديثة {s.rsi_min:.0f})")
    if cd["breakout"]:
        if cd.get("vol_ok") is False:
            wait.append("فوليوم أعلى عند اختراق خط العنق (الاختراق بدون فوليوم ضعيف)")
    elif not (ctx.vol_improving or ctx.sell_drying):
        wait.append("تأكيد الفوليوم: تحسن مع الارتداد أو ضعف فوليوم البيع")
    distinct = len(np.unique(np.round(ctx.c[-20:], 4)))
    if distinct < C.MIN_DISTINCT_CLOSES:
        s.flags.append(f"حركة السعر متقطعة ({distinct} أسعار إغلاق مختلفة في 20 يوم) — قراءة الشموع غير موثوقة")
        wait.append("سعر يتحرك بتكات كبيرة — الشموع والمؤشرات غير موثوقة على هذا السهم")
    if not ctx.former_runner:
        wait.append("ليس Former Runner — ما سبق انفجر خلال سنة (فيصل يبحث عن سهم يقدر يتحرك بعنف)")
    if cond["pressure"][0] < 0.5:
        wait.append("سهم الارتكاز يحتاج هبوطاً/ضغط بيع سابقاً (25%+)")
    if cond["macd"][0] < 1.0:
        wait.append("تأكيد MACD (تقاطع إيجابي أو تحسن يومين متتاليين)")
    if not cd["breakout"] and cond["candle"][0] < 1.0 and cd["code"] in ("SWEEP", "BOUNCE"):
        wait.append("شمعة انعكاس واضحة عند الدعم (مطرقة/ابتلاعية بحجم واضح)")
    if ctx.pump_month is not None:
        j, hi_, cl_, vm_ = ctx.pump_month
        days = ctx.n - 1 - j
        s.flags.append(f"شمعة ضخ/تصريف قبل {days} يوم (طار إلى {hi_:.3g} وأغلق {cl_:.3g} بفوليوم {vm_:.0f}x)")
        wait.append("مرور وقت كافٍ بعد شمعة الضخ/التصريف (سلوك قروبات الوهم)")
    if rr is None:
        wait.append("هدف علوي واضح")
    elif rr < C.MIN_RR_T1:
        wait.append(f"الدخول تحت مقاومة مباشرة ({t1[0]:.2f}) — انتظر اختراقها والثبات")
    elif rr_best < C.MIN_RR_SPACE:
        wait.append("مساحة صعود أكبر حتى المقاومات القادمة")
    if risk > C.MAX_RISK_PIVOT:
        wait.append(f"الوقف بعيد ({risk*100:.0f}%) — انتظر نزول السعر قرب الدعم")
    # "لا تطارد إذا لم ينفذ لك": جاهز فقط إذا كان السعر الآن داخل منطقة الطلب أو قريباً منها
    if not cd["breakout"] and cd["confirmed"] and price > entry_high * (1 + C.MAX_ABOVE_ENTRY):
        wait.append(f"السعر ابتعد {(price / entry_high - 1) * 100:.0f}% فوق منطقة الطلب — "
                    f"ضع Limit عند {entry_low:.2f}–{entry_high:.2f} ولا تطارد")
    if not cd["breakout"] and ctx.gain_from_low10 >= C.CHASE_GAIN:
        pct = ctx.gain_from_low10 * 100
        wait.append(f"صعد {pct:.0f}% من القاع — لا مطاردة، انتظر Retest أو سحب سيولة")
    if ctx.pump_recent or ctx.distribution_recent:
        s.flags.append("شمعة ضخ/تصريف حديثة (سلوك قروبات وهم)")
        wait.append("هدوء بعد شمعة الضخ وتكوين دعم جديد")
    if ctx.fail_at_ma20 and not cd["breakout"]:
        s.flags.append("فشل عند MA20")
    if score < C.MIN_SCORE_READY:
        wait.append(f"اكتمال الشروط الأساسية ({score:.1f} من 6، المطلوب {C.MIN_SCORE_READY:.1f})")

    s.waiting_for = list(dict.fromkeys(wait))
    s.status = READY if not s.waiting_for else WATCH

    # هل يستحق الظهور في قائمة المراقبة؟
    if s.status == WATCH:
        close_to_trigger = abs(price / cd["trigger"] - 1) <= 0.15
        pressure = s.rsi_min <= 35 and ctx.drawdown >= C.DRAWDOWN_MED
        if not (close_to_trigger and pressure and score >= 2.5):
            return None
    s.rank = score + (2.0 if s.status == READY else 0.0) + (1.0 if ctx.former_runner else -1.0) \
        - min(abs(price / cd["trigger"] - 1), 0.3) * 5
    return s
