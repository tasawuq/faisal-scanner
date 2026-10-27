"""
تحليل سهم الارتكاز حسب منهج فيصل — بعد إعادة قراءة الملف كاملاً (27 أكتوبر، القواعد في method.py).

- التحليل على **الشموع اليومية المكتملة فقط** (ص5، ص54: فريمات يومية/أسبوعية) — شمعة اليوم الجارية ما تُحسب،
  وفريم 4 ساعات (ص90 «اقربها 4 ساعات») يُضاف في main.py من شموع اليوم المكتملة.
- لا يوجد «جاهز». الإشارة الوحيدة = «اكتملت شروط ص86» (قاع ← إعادة اختبار وفشل الكسر ← ثبات ← شمعة انعكاس
  بفوليوم يتحسن + المتوسطات + RSI + MACD)، وبدون مقاومة مباشرة (ص88)، وبعدها طلبات Limit مقسّمة داخل المنطقة.
- كل سهم غير مكتمل: «لماذا غير جاهز» (الشروط الناقصة بصفحاتها) و«متى يكون جاهزاً» (ص70).
"""
from dataclasses import dataclass, field, asdict
import math
import numpy as np
import pandas as pd

from . import config as C
from . import method as M
from .indicators import sma, rsi, macd, atr
from .levels import (swing_lows, swing_highs, build_zones, count_visits, resistance_levels, candle, candle_name,
                     Zone)

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
    # 🚩 ص79 مثال 4 (أيقونة العلم): "اختراق مدروس بعد قاعدة — كوّن قاعدة، اخترق مقاومة قريبة، رجع لاختبارها ونجح،
    # RSI كان 27 قبل الانطلاق → دخول أقوى بعد إعادة الاختبار" + ص68 (هبوط-قاع-كسر مقاومة-Retest-ثبات-دخول) + ص86 دخول 3
    "BASE_RETEST": "🚩 اختراق مدروس بعد قاعدة + إعادة اختبار ناجحة",
    "BASE_BREAK": "🚩 اختراق مقاومة قريبة بعد قاعدة — بانتظار إعادة الاختبار",
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

        # Former Runner (ص10: «سبق انفجر 100% أو أكثر») — كل التاريخ المتوفر
        run = M.runner(df)
        self.runner_pct = float(run.get("pct") or 0.0)
        self.former_runner = bool(run.get("is_runner"))
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


# ====================================================================== 🚩 base breakout + retest
def detect_base_breakout(ctx: Ctx):
    """ص79 مثال 4 (🚩) + ص68 + ص86 (دخول 3): هبوط → قاع/قاعدة بضغط RSI → اختراق مقاومة قريبة → رجوع لاختبارها
    والثبات فوقها. يرجع آخر نموذج صالح أو None."""
    n = ctx.n
    start = max(0, n - 60)
    if n - 4 <= start:
        return None
    lo_i = start + int(np.argmin(ctx.l[start:n - 3]))
    base_low = float(ctx.l[lo_i])
    # الهبوط قبل القاعدة: قمة الـ 60 يوم السابقة للقاع أعلى منه بـ 25%+ (نموذج الأسهم الهابطة ص68)
    pre = ctx.h[max(0, lo_i - 60):lo_i + 1]
    if not len(pre) or float(np.max(pre)) < base_low * (1 + C.DRAWDOWN_MED):
        return None
    highs = [i for i in swing_highs(ctx.h, k=2) if max(start, lo_i - 15) <= i < n - 2]
    # المقاومة القريبة = سقف القاعدة: أعلى قمة محلية (ضمن +6%..+45% من القاع) تم كسرها بإغلاق بعد تكوّنها
    broken = []
    for i in highs:
        res = float(ctx.h[i])
        if not (base_low * 1.06 <= res <= base_low * 1.45):
            continue
        j = next((k for k in range(max(i, lo_i) + 1, n) if ctx.c[k] > res * 1.01), None)
        if j is not None:
            broken.append((res, i, j))
    for res, i, j in sorted(broken, reverse=True)[:1]:
        held = bool(np.all(ctx.c[j:] >= res * 0.97)) and ctx.c[-1] >= res
        if not held:
            return {"trap": True, "res": res, "brk": j}  # اخترق ثم رجع تحت = اختراق وهمي (Bull Trap)
        # ص68/ص86: «يرجع لاختبار الدعم الجديد» = القاع يوصل المستوى نفسه (هامش 1%) — IDN 27 أكتوبر: قيعان 2.40 فوق 2.34
        # بـ 2.6% ما هي إعادة اختبار
        k_rt = next((k for k in range(j + 1, n) if ctx.l[k] <= res * 1.01), None)
        vol_ok = bool(ctx.v[j] >= 1.2 * max(ctx.avg_vol, 1)
                      or (j == n - 1 and ctx.v_last_adj >= 1.2 * max(ctx.avg_vol, 1)))
        out = {"res": res, "res_idx": i, "brk": j, "low_idx": lo_i, "base_low": base_low, "vol_ok": vol_ok,
               "retest": k_rt}
        if k_rt is not None:
            # نجاح إعادة الاختبار: إغلاق شمعة الاختبار فوق المقاومة تقريباً + آخر شمعة ثابتة (إغلاق فوق المقاومة
            # وفي النصف العلوي أو خضراء) — "إغلاق شموع فوق الدعم / شموع رفض عند الدعم" ص86
            rng = max(ctx.h[-1] - ctx.l[-1], 1e-9)
            out["retest_ok"] = bool(ctx.c[k_rt] >= res * 0.99 and ctx.c[-1] >= res
                                    and (ctx.c[-1] >= ctx.o[-1] or (ctx.c[-1] - ctx.l[-1]) / rng >= 0.5))
            out["retest_low"] = float(np.min(ctx.l[k_rt:]))
        return out
    return None




# ====================================================================== main entry
ALLOWED_READY = ("SWEEP", "BOUNCE", "AT_SUPPORT", "W_RETEST", "BASE_RETEST")  # دخول 1/2/3 في ص86 — لا شراء للاختراق نفسه (ص20)
BREAKOUT_SETUPS = ("W_BREAK", "W_RETEST", "BASE_BREAK", "BASE_RETEST")


def analyze_pivot(ticker: str, df: pd.DataFrame, session_frac: float = 1.0, last_bar_partial: bool = False,
                  short_pct=None):
    """last_bar_partial=True: آخر شمعة = يوم جاري (غير مكتمل) → التحليل على الشموع المكتملة، والسعر الحالي للعرض
    وللفحص (تحت الوقف / مطاردة) فقط."""
    if df is None or len(df) < 61:
        return None
    df = df.dropna(subset=["open", "high", "low", "close"])
    px_now = float(df["close"].iloc[-1])
    today = None
    if last_bar_partial:
        today = {k: float(df[k].iloc[-1]) for k in ("open", "high", "low", "close", "volume")}
        df = df.iloc[:-1]
    if len(df) < 60:
        return None
    ctx = Ctx(df, 1.0, False)
    price = ctx.price
    ma_vals = {"MA20": ctx.ma[20][-1], "MA30": ctx.ma[30][-1], "MA50": ctx.ma[50][-1]}
    cands = []

    # ---------------- الدعم وسحب السيولة (دخول 1 و2 في ص86)
    sup = find_support(ctx)
    if sup is not None:
        z, interacting = sup
        recent_low = float(np.min(ctx.l[-3:]))
        swept = recent_low < z.low * 0.995 and ctx.c[-1] >= z.low
        reversal = any(info["bullish_reversal"] and ctx.l[j] <= z.high * 1.03 for j, info in ctx.candles)
        if ctx.c[-1] < z.low * 0.99 and ctx.c[-2] < z.low * 0.99:
            code = "BROKE_SUPPORT"  # كسر وبقي تحته = فشل (ص17) — يبقى بالقائمة بانتظار الاسترداد أو المنطقة الأعمق
        elif interacting and swept:
            code = "SWEEP"
        elif interacting and reversal:
            code = "BOUNCE"
        elif price < z.low:
            code = "BROKE_SUPPORT"
        else:
            code = "AT_SUPPORT"
        cands.append(dict(code=code, zone=(z.low, z.high), trigger=z.low, touches=z.touches, breakout=None,
                          is_bottom=z.is_bottom))

    # ---------------- نموذج W (ص25، ص51)
    w = detect_w(ctx)
    if w is not None:
        neck, b = w["neck"], w["b"]
        brk = next((j for j in range(b + 1, ctx.n) if ctx.c[j] > neck), None)
        if brk is None:
            if price >= w["low2"] * 1.02 and price >= neck * 0.85:
                cands.append(dict(code="W_FORMING", zone=(min(w["low1"], w["low2"]), max(w["low1"], w["low2"]) * 1.02),
                                  trigger=neck, touches=2, breakout=None, neck=neck))
        else:
            held = bool(np.all(ctx.c[brk:] >= neck * 0.98)) and ctx.c[-1] >= neck
            retest = brk + 1 < ctx.n and np.min(ctx.l[brk + 1:]) <= neck * 1.01 and held
            brk_vol_ok = bool(ctx.v[brk] >= 1.2 * max(ctx.avg_vol, 1))
            if held:
                last = ctx.n - 1
                rng = max(ctx.h[last] - ctx.l[last], 1e-9)
                hold_bar = ctx.c[last] >= ctx.o[last] or (ctx.c[last] - ctx.l[last]) / rng >= 0.5
                cands.append(dict(code="W_RETEST" if retest else "W_BREAK", zone=(neck * 0.98, neck * 1.01),
                                  trigger=neck, touches=2, neck=neck,
                                  breakout={"res": neck, "retest_ok": bool(retest and hold_bar), "vol_ok": brk_vol_ok,
                                            # الدعم الحرج: قاع إعادة الاختبار إذا صارت، وإلا تحت خط العنق مباشرة
                                            "retest_low": (float(np.min(ctx.l[brk + 1:])) if retest
                                                           else neck * 0.97)}))
            elif ctx.c[-1] < neck and brk >= ctx.n - 10:
                ctx.bull_trap = True

    # ---------------- 🚩 اختراق مدروس بعد قاعدة + إعادة اختبار (ص79 مثال 4، ص68، ص86 دخول 3)
    bb = detect_base_breakout(ctx)
    if bb is not None and bb.get("trap"):
        ctx.bull_trap = True
    elif bb is not None:
        res = bb["res"]
        ok = bool(bb.get("retest") is not None and bb.get("retest_ok"))
        cands.append(dict(code="BASE_RETEST" if ok else "BASE_BREAK", zone=(res * 0.98, res * 1.01), trigger=res,
                          touches=2, neck=res,
                          breakout={"res": res, "retest_ok": ok, "vol_ok": bb["vol_ok"],
                                    "retest_low": bb.get("retest_low") or res * 0.97}))

    if not cands:
        return None
    results = [_evaluate(ticker, ctx, cd, ma_vals, px_now, today) for cd in cands]
    results = [r for r in results if r is not None]
    if not results:
        return None
    results.sort(key=lambda r: (r.status == READY, r.rank + (0.25 if r.setup == "BASE_RETEST" else 0.0)), reverse=True)
    best = results[0]
    if getattr(ctx, "bull_trap", False):
        best.flags.append("اختراق وهمي سابق (Bull Trap — ص19، ص57)")
    return best


def _evaluate(ticker, ctx: Ctx, cd: dict, ma_vals: dict, px_now: float, today: dict = None):
    df = ctx.df
    z_low, z_high = float(cd["zone"][0]), float(cd["zone"][1])
    if z_high < z_low:
        z_low, z_high = z_high, z_low
    p86 = M.p86_daily(df, z_low, z_high, cd.get("breakout"))
    if cd["code"] == "W_FORMING":
        p86["critical"] = float(z_low)
    crit = min(float(p86["critical"]), z_low)  # الدعم الحرج تحت المنطقة دائماً (BJDX: المنطقة 1.70–1.95 والحرج 1.67)
    p86["critical"] = crit
    lad = M.ladder(z_low, z_high, crit)
    if lad["avg"] <= lad["stop"]:
        return None
    # المقاومة المباشرة = أول مستوى فوق الطلبات وفوق آخر إغلاق (المستوى اللي السعر أغلق فوقه انكسر وما عاد سقفاً)
    top = max(lad["orders"][0], float(ctx.c[-1]))
    res = resistance_levels(df, top, C.LOOKBACK_LEVELS, C.SWING_K, ma_vals, floor_pct=0.005)
    near = [float(x) for x in ctx.h[-5:] if x > top * 1.005]  # قمم آخر 5 أيام فوق = مقاومة مباشرة (ص88)
    if today and today.get("high") and today["high"] > top * 1.005:
        near.append(float(today["high"]))  # قمة اليوم الجاري (NERV 27 أكتوبر: 2.335 فوق الدخول 2.25 مباشرة)
    if near:
        nr = min(near)
        if not res or nr < res[0][0] * 0.99:
            res = [(nr, "قمة يوم قريبة")] + list(res)
    t1 = res[0] if res else None
    t2 = res[1] if len(res) > 1 else None

    s = Signal(ticker=ticker, setup=cd["code"], setup_ar=SETUP_AR[cd["code"]], price=float(px_now))
    s.support_low, s.support_high, s.support_touches = z_low, z_high, int(cd.get("touches") or 0)
    s.trigger_level = float(cd["trigger"])
    s.entry_low, s.entry_high = float(lad["orders"][2]), float(lad["orders"][0])
    if t1:
        s.t1, s.t1_label = float(t1[0]), t1[1]
    if t2:
        s.t2, s.t2_label = float(t2[0]), t2[1]
    s.rsi_now = round(float(ctx.rsi[-1]), 1)
    s.rsi_min = round(float(np.nanmin(ctx.rsi[-M.BASE_WINDOW:])), 1)
    s.ma_state, s.macd_state = ctx.ma_state, ctx.macd_state
    s.volume_note = ctx.volume_note
    run = M.runner(df)
    dbl = M.just_doubled(df if today is None else pd.concat([df, pd.DataFrame([today], index=[df.index[-1] + pd.Timedelta(days=1)])]))
    s.meta = {"avg_volume": ctx.avg_vol, "drawdown": round(float(ctx.drawdown), 3),
              "gain_from_low10": round(float(ctx.gain_from_low10), 3), "higher_lows": ctx.higher_lows,
              "runner": run, "today": today, "zone": [z_low, z_high], "breakout_setup": bool(cd.get("breakout")),
              "last_close": float(ctx.c[-1]), "last_date": str(df.index[-1])[:10],
              "neck": cd.get("neck")}
    if run.get("is_runner"):
        s.tags.insert(0, f"Former Runner: صعد {run['pct'] * 100:.0f}% من {M.rnd(run['from'])} ({run['from_date']}) "
                         f"إلى {M.rnd(run['to'])} ({run['to_date']})")
    if ctx.higher_lows:
        s.tags.append("قيعان أعلى")
    if ctx.sell_drying:
        s.tags.append("فوليوم البيع يضعف")
    if cd.get("is_bottom"):
        s.tags.append("قاع الفترة (قاع سنة)")

    static = []
    if dbl:
        s.flags.append(f"توه مدبل: من {M.rnd(dbl[0])} إلى {M.rnd(dbl[1])}")
        static.append(f"توه مدبل (من {M.rnd(dbl[0])} إلى {M.rnd(dbl[1])}) — «فيه خطوره… غير امن الان» (ص59)")
    if ctx.pump_recent or ctx.distribution_recent:
        s.flags.append("شمعة ضخ/تصريف حديثة (سلوك قروبات وهم)")
        static.append("شمعة ضخ/تصريف حديثة — سلوك قروبات الوهم (ص10، ص20)")
    if ctx.pump_month is not None:
        j, hi_, cl_, vm_ = ctx.pump_month
        s.flags.append(f"شمعة ضخ/تصريف قبل {ctx.n - 1 - j} يوم (طار إلى {hi_:.3g} وأغلق {cl_:.3g} بفوليوم {vm_:.0f}x)")
        static.append("مرور وقت كافٍ بعد شمعة الضخ/التصريف (ص10، ص20)")
    distinct = len(np.unique(np.round(ctx.c[-20:], 4)))
    if distinct < C.MIN_DISTINCT_CLOSES:
        s.flags.append(f"حركة السعر متقطعة ({distinct} أسعار إغلاق مختلفة في 20 يوم)")
        static.append("السعر يتحرك بتكات كبيرة — الشموع والمؤشرات غير موثوقة")
    if ctx.fail_at_ma20 and not cd.get("breakout"):
        s.flags.append("رفض عند MA20 (ص64)")
    if cd["code"] in ("W_BREAK", "BASE_BREAK"):
        static.append(f"اخترق {M.rnd(cd['trigger'])} — لا شراء للاختراق نفسه، انتظر رجوعه لاختباره والثبات (ص20، ص68)")
    if cd["code"] == "W_FORMING":
        static.append(f"نموذج W ما اكتمل: كسر خط العنق {M.rnd(cd['trigger'])} والثبات (ص25)")
    s.meta["static_wait"] = static
    s.meta["p86"] = {k: (v if k != "checks" else {kk: [bool(x[0]), x[1]] for kk, x in v.items()}) for k, v in p86.items()}
    s.meta["when_ready"] = M.when_ready(cd["code"], z_low, z_high, crit, cd.get("neck"), s.t1)
    apply_gate(s)

    # القائمة: متابعة الأسهم القريبة من منطقتها (اجتهاد: ضمن 20%) واللي اكتمل منها 3 شروط على الأقل
    met = len(M.summary(s.meta["p86"]["checks"])[0])
    if s.status == WATCH:
        dist = px_now / ((z_low + z_high) / 2) - 1
        if not (-0.2 <= dist <= 0.2 and met >= 3):
            return None
    s.rank = met + (2.0 if s.status == READY else 0.0) + (1.0 if run.get("is_runner") else 0.0) \
        - min(abs(px_now / s.trigger_level - 1), 0.3) * 5
    return s


def apply_gate(s: Signal):
    """يبني «لماذا غير جاهز» من شروط ص86 الناقصة + الموانع (مقاومة مباشرة، الخسارة، المطاردة، تحت الوقف، السلوك الخطر)
    ويحدد الحالة. يُستدعى بعد التحليل اليومي، وبعد ترقية فريم 4 ساعات (main.py)."""
    p = s.meta["p86"]
    z_low, z_high = s.meta["zone"]
    lad = M.ladder(z_low, z_high, p["critical"])
    s.meta["ladder"] = lad
    s.entry_low, s.entry_high = float(lad["orders"][2]), float(lad["orders"][0])
    s.stop = float(lad["stop"])
    s.risk_pct = round(float(lad["risk"]), 4)
    s.rr = round((s.t1 - lad["avg"]) / (lad["avg"] - lad["stop"]), 2) if s.t1 else None
    checks = p["checks"]
    s.conditions = {k: {"score": 1.0 if v[0] else 0.0, "note": v[1]} for k, v in checks.items()}
    s.score = float(sum(1 for v in checks.values() if v[0]))
    s.candle = checks.get("candle", [False, ""])[1] if checks.get("candle", [False])[0] else ""
    wait = [f"{M.NAME[k]}: {checks[k][1]} ({M.PAGE[k]})" for k in M.summary(checks)[1]]
    wait += list(s.meta.get("static_wait") or [])
    if s.t1 is not None and s.rr is not None and s.rr <= 1.0:
        wait.append(f"تحت مقاومة مباشرة {M.rnd(s.t1)} (+{(s.t1 / lad['avg'] - 1) * 100:.1f}%) والخسارة للوقف "
                    f"−{lad['risk'] * 100:.1f}% — العائد ما يتعدى المخاطرة (ص82، ص88)")
    elif s.t1 is None:
        wait.append("ما فيه هدف/مقاومة واضحة فوق لحساب العائد")
    if lad["risk"] > M.MAX_LOSS:
        wait.append(f"الخسارة من متوسط الطلبات للوقف {lad['risk'] * 100:.0f}% أكبر من 15% (ص54)")
    px = s.price
    td = s.meta.get("today") or {}
    if td.get("low") is not None and td["low"] < lad["stop"] and px >= lad["stop"]:
        # LGHL 27 أكتوبر: نزل اليوم إلى 2.00 تحت الوقف 2.24 ثم رجع 2.26 ونبّه على شمعة أمس — الدعم انكسر (ص17، ص89)
        wait.append(f"نزل اليوم إلى {M.rnd(td['low'])} تحت الدعم الحرج {M.rnd(lad['critical'])} — يا سحب سيولة يا كسر: "
                    f"ينتظر إغلاق الشمعة (يومي/4 ساعات) فوق الدعم، والوقف يصير تحت قاع السحب (ص17)")
    if px < lad["stop"]:
        wait.append(f"السعر الآن {M.rnd(px)} تحت الدعم الحرج {M.rnd(lad['critical'])} — ينتظر الاسترداد (ص17)")
    elif px < lad["orders"][-1]:
        # طلب Limit فوق السعر = شراء من العرض فوراً (ص57) — والسعر تحت المنطقة يعني الدعم ما استُرد
        wait.append(f"السعر الآن {M.rnd(px)} تحت منطقة الطلبات {M.rnd(lad['orders'][-1])} — ينتظر استردادها (ص17، ص57)")
    elif px / p["critical"] - 1 >= M.CHASE_GAIN:
        wait.append(f"السعر صعد {(px / p['critical'] - 1) * 100:.0f}% فوق الدعم الحرج — لا مطاردة (ص67، ص88)")
    for fl in s.flags:
        if "صيني" in fl:
            wait.append("سهم صيني عالي التذبذب — «وهذا وحده يجعله خطيراً» (ص70، ص90): متابعة بدون إشارة")
        if "سيولة ضعيفة" in fl or "الفلووت غير معروف" in fl or "الفلووت أكبر" in fl:
            wait.append(fl)
    s.waiting_for = list(dict.fromkeys(wait))
    s.status = READY if (not s.waiting_for and s.setup in ALLOWED_READY) else WATCH
    if s.status == WATCH and not s.waiting_for:
        s.waiting_for = ["نموذج الدخول ما اكتمل"]
    return s
