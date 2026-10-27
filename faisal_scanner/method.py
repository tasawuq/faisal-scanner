"""
منهج فيصل حرفياً — بعد إعادة قراءة الملف كاملاً (27 أكتوبر 2026، بموافقة المستخدم على النقاط السبع).
كل رقم هنا مكتوب في الملف، ومعه رقم الصفحة. الأرقام اللي مو من الملف مكتوب جنبها «اجتهاد» وسببه.

1) الاختيار (ص6، ص8، ص56، ص59، ص61، ص90–94): ناسداك، قيمة سوقية تحت 300M، فلووت/عدد أسهم مثل أسهم فيصل (≤8M)، فوليوم فوق 100K،
   والأولوية لعدد أسهم صغير جداً + تقسيم عكسي حديث + شورت قليل + سهم سبق صعد 100%+، و«توه مدبل» غير آمن الآن.
2) لا «جاهز» ولا دخول بالسعر اللحظي (ص54، ص70): كل سهم «لماذا غير جاهز» و«متى يكون جاهزاً».
3) الفريم يومي و4 ساعات (ص90: «قراءة الشموع تتم ع جميع الفريمات، اقربها 4 ساعات»).
4) الإشارة فقط عند الترتيب الكامل (ص86): قاع ← إعادة اختبار يفشل فيها الكسر ← ثبات ← شمعة انعكاس مع فوليوم يتحسن،
   + الشروط الستة «قبل الدخول تأكد من اكتمال الشروط» (ص86)، ولا دخول تحت مقاومة مباشرة (ص67، ص88).
5) الدخول: طلبات Limit مقسّمة 3 داخل منطقة الطلب (ص15–16، ص18، ص57)، والخروج بكسر الدعم الحرج (BJDX ص57).
"""
from datetime import datetime, timezone, timedelta, time

import numpy as np
import pandas as pd

from .indicators import sma, rsi, macd
from .levels import candle, candle_name

# ------------------------------------------------------------------ الأرقام (كلها من الملف)
EXCHANGE = "NASDAQ"          # ص6، ص7، ص61 (Exchange NASDAQ)
MAX_PRICE = 5.0              # ص6 Price Under $5
MAX_MARKET_CAP = 300e6       # ص6 «Market Cap Micro أو Under 300M»
# المستخدم 27 أكتوبر «خل نفس فيصل بالضبط حتى عدد الاسهم»: نفس مدى أسهم فيصل الفعلية بدل Float Under 20M (ص61):
# فرز الرسائل ص56/ص59: عدد الأسهم 849 ألف – 3.05M؛ قائمة الارتكاز الأسبوعية ص90–94: الفلوت 541 ألف – 7.84M (BNKK)
# (MWC 26.68M طرح جديد — نوع مختلف، والطرح الجديد خارج الفحص لأن تاريخه أقل من 60 يوم). وما فيه حد أدنى للقيمة السوقية:
# كل أسهمه بهذا العدد قيمتها تحت 50M.
MAX_FLOAT = 8e6
MIN_AVG_VOLUME = 100_000     # ص6، ص61 Average Volume Over 100K
RUNNER_MIN = 1.00            # ص10 «Former Runner: سبق انفجر 100% أو أكثر»
RUNNER_WINDOW = 21           # اجتهاد: «خلال شهر» — LFS ص32 «صعد معنا قبل اقل من شهر من 1.60 ل 5»
JUST_DOUBLED = 1.00          # ص59 «توه مدبل فيه خطوره» / «توه صاعد من سنتات ل 3 غير امن الان»
JUST_DOUBLED_BARS = 10       # اجتهاد: «توه» = آخر أسبوعين تداول
RECENT_SPLIT_DAYS = 210      # أمثلة الملف: تقسيمات أسهم فيصل خلال أيام إلى ~7 أشهر (ص56، ص59، ص90–93)
RSI_PRESSURE = 27            # ص65، ص78، ص86 «RSI في التشبع البيعي 23 - 27» (وصلها أو أعمق)
CHASE_GAIN = 0.20            # ص67، ص88 «مطاردة السهم بعد ارتفاع 20-50%»
MAX_LOSS = 0.15              # ص54 «نسبة الخطا الاسبوعيه الاصل لاتتجاوز خساره 15٪»
BASE_WINDOW = 40             # اجتهاد: نافذة القاعدة على اليومي (~شهرين) لقراءة القاع وإعادة الاختبار وRSI
REGULAR = (time(9, 30), time(16, 0))
H4_SPLIT = time(13, 30)      # شمعة 4 ساعات للجلسة الرسمية: 9:30–13:30 ثم 13:30–16:00

PAGE = {"ma": "ص85/ص86", "rsi": "ص65/ص86", "macd": "ص85/ص86", "support": "ص86", "retest": "ص86/ص17",
        "volume": "ص85/ص86", "candle": "ص82/ص86"}
NAME = {"ma": "تحت المتوسطات 20/30/50 ثم تماسك", "rsi": "RSI وصل 23–27", "macd": "MACD يتحسن أو تقاطع",
        "support": "دعم/قاع واضح", "retest": "اختبار الدعم وفشل الكسر", "volume": "فوليوم يتحسن مع الارتداد",
        "candle": "شمعة انعكاس عند الدعم"}
ORDER = ("ma", "rsi", "macd", "support", "retest", "volume", "candle")


def tick(px: float) -> float:
    return 0.01 if px >= 1 else 0.0001


def rnd(x: float) -> float:
    return round(float(x), 2 if x >= 1 else 4)


# ------------------------------------------------------------------ الاختيار
def runner(df: pd.DataFrame) -> dict:
    """أكبر صعود من قاع إلى قمة خلال شهر (21 يوم تداول) في كل التاريخ المتوفر (ص10، ص31–32).
    يرجع {pct, from, from_date, to, to_date, is_runner}."""
    if df is None or len(df) < 5:
        return {}
    lo, hi = df["low"].values.astype(float), df["high"].values.astype(float)
    n = len(lo)
    best = (0.0, 0, 0)
    for i in range(n - 1):
        if not np.isfinite(lo[i]) or lo[i] <= 0:
            continue
        j_end = min(n, i + 1 + RUNNER_WINDOW)
        seg = hi[i + 1:j_end]
        if not len(seg):
            continue
        j = i + 1 + int(np.nanargmax(seg))
        r = hi[j] / lo[i] - 1
        if r > best[0]:
            best = (r, i, j)
    r, i, j = best
    idx = df.index
    return {"pct": float(r), "from": float(lo[i]), "from_date": str(idx[i])[:10], "to": float(hi[j]),
            "to_date": str(idx[j])[:10], "is_runner": bool(r >= RUNNER_MIN)}


def just_doubled(df: pd.DataFrame):
    """ص59: «توه مدبل» = خلال آخر أسبوعين القمة ضعف القاع اللي قبلها أو أكثر → غير آمن الآن. يرجع (قاع، قمة) أو None."""
    if df is None or len(df) < 3:
        return None
    lo, hi = df["low"].values.astype(float), df["high"].values.astype(float)
    n = len(lo)
    for j in range(max(1, n - JUST_DOUBLED_BARS), n):
        base = float(np.nanmin(lo[max(0, j - JUST_DOUBLED_BARS):j]))
        if base > 0 and hi[j] / base - 1 >= JUST_DOUBLED:
            return base, float(hi[j])
    return None


def split_info(fund: dict, df: pd.DataFrame = None, now=None) -> dict:
    """آخر تقسيم + سعر الافتتاح بعده + أعلى سعر بعده (قالب فيصل ص90–94: «قسم 6/4 افتتاح 3.72 - اعلى 4.58»)."""
    f = fund or {}
    fac, dt = f.get("lastSplitFactor"), f.get("lastSplitDate")
    if not fac or not dt:
        return {}
    try:
        a, b = [float(x) for x in str(fac).split(":")]
        when = datetime.fromtimestamp(float(dt), tz=timezone.utc)
    except Exception:  # noqa
        return {}
    now = now or datetime.now(timezone.utc)
    out = {"factor": str(fac), "date": when.strftime("%Y-%m-%d"), "reverse": a < b, "days": (now - when).days}
    out["recent"] = bool(out["reverse"] and out["days"] <= RECENT_SPLIT_DAYS)
    if df is not None and len(df):
        d = df[[ts.date() >= when.date() for ts in df.index]]
        if len(d) and d.index[0].date() <= (when + timedelta(days=5)).date():
            out["open_after"] = float(d["open"].iloc[0])
            out["high_after"] = float(d["high"].max())
    return out


COUNTRY_AR = {"united states": "أمريكي", "china": "صيني", "hong kong": "صيني (هونغ كونغ)", "israel": "إسرائيلي",
              "japan": "ياباني", "singapore": "سنغافوري", "canada": "كندي", "united kingdom": "بريطاني",
              "taiwan": "تايواني", "malaysia": "ماليزي", "australia": "أسترالي", "germany": "ألماني",
              "switzerland": "سويسري", "netherlands": "هولندي", "greece": "يوناني", "cayman islands": "جزر كايمان",
              "british virgin islands": "جزر العذراء البريطانية", "south korea": "كوري", "india": "هندي",
              "ireland": "إيرلندي", "france": "فرنسي", "sweden": "سويدي", "denmark": "دنماركي", "brazil": "برازيلي",
              "argentina": "أرجنتيني", "mexico": "مكسيكي", "thailand": "تايلندي", "vietnam": "فيتنامي",
              "united arab emirates": "إماراتي", "cyprus": "قبرصي", "bermuda": "برمودا", "monaco": "موناكو"}


def country_ar(c: str) -> str:
    c = (c or "").strip()
    return COUNTRY_AR.get(c.lower(), c) if c else ""


# ------------------------------------------------------------------ مساعدات
def _ind(df):
    c = df["close"].astype(float)
    ma = {p: sma(c, p).values for p in (20, 30, 50)}
    r = rsi(c, 14).values
    line, sig, hist = macd(c, 12, 26, 9)
    return ma, r, line.values, sig.values, hist.values


def macd_ok(line, sig, hist) -> tuple:
    n = len(hist)
    cross = any(line[j - 1] < sig[j - 1] and line[j] >= sig[j] for j in range(max(1, n - 3), n))
    if cross:
        return True, "تقاطع إيجابي حديث"
    if n >= 3 and hist[-1] > hist[-2] > hist[-3]:
        return True, "الهيستوجرام يتحسن"
    if line[-1] > sig[-1]:
        return True, "الخط فوق الإشارة"
    return False, "سلبي — ما بدأ يتحسن"


def episodes(low, zone_high, start, end):
    """زيارات منطقة الدعم: فترات متتالية لمس فيها القاع المنطقة (يفصلها يومان على الأقل خارجها)."""
    eps, cur, gap = [], None, 99
    for i in range(start, end):
        inside = low[i] <= zone_high
        if inside:
            if cur is not None and gap < 2:
                cur[1] = i
            else:
                cur = [i, i]
                eps.append(cur)
            gap = 0
        else:
            gap += 1
    return eps


# ------------------------------------------------------------------ شروط ص86 على اليومي
def p86_daily(df: pd.DataFrame, z_low: float, z_high: float, breakout: dict = None) -> dict:
    """الشروط الستة «قبل الدخول تأكد من اكتمال الشروط» (ص86) + شمعة الانعكاس عند الدعم (ص82، ص86) على شموع يومية
    مكتملة فقط. breakout: لنموذج الدخول 3 (اختراق مقاومة قريبة ثم إعادة اختبار) — {"res", "retest_ok", "vol_ok",
    "retest_low"}. يرجع {checks:{k:(ok, note)}, critical, complete, bar_date}."""
    n = len(df)
    o, h, l, c, v = (df[x].values.astype(float) for x in ("open", "high", "low", "close", "volume"))
    ma, r, line, sig, hist = _ind(df)
    st = max(0, n - BASE_WINDOW)
    zh = z_high * 1.03  # «قرب الدعم»: لمس المنطقة أو أعلاها بـ 3% (اجتهاد لتسامح الشمعة)
    out = {}
    mins = np.fmin(np.fmin(ma[20], ma[30]), ma[50])
    below = any(np.isfinite(mins[i]) and c[i] < mins[i] for i in range(st, n))
    eps = [] if breakout else episodes(l, zh, st, n)
    broke = False
    if breakout:
        crit = float(breakout.get("retest_low") or z_low)
    else:
        last_ep = eps[-1] if eps else None
        crit = float(min(z_low, np.min(l[last_ep[0]:last_ep[1] + 1]))) if last_ep else float(z_low)
        if last_ep:
            # ص17: «إذا كسر الدعم وبقي تحته أو فشل في الرجوع فوقه → ليس سحب سيولة بل ضعف» (إغلاقان متتاليان تحته)
            broke = any(c[i] < z_low * 0.99 and c[i + 1] < z_low * 0.99 for i in range(last_ep[0], n - 1))
    holding = bool(np.all(c[-3:] >= crit))
    out["ma"] = (below and holding,
                 ("أغلق تحت المتوسطات خلال القاعدة" if below else "ما كان تحت المتوسطات الثلاثة خلال القاعدة")
                 + ("، وثابت فوق الدعم الحرج آخر 3 أيام" if holding else f"، لكن أغلق تحت الدعم الحرج {rnd(crit)}"))
    rmin = float(np.nanmin(r[st:])) if n > st else float("nan")
    out["rsi"] = (bool(np.isfinite(rmin) and rmin <= RSI_PRESSURE), f"أدنى RSI {rmin:.0f} (الحالي {r[-1]:.0f})")
    mok, mnote = macd_ok(line, sig, hist)
    out["macd"] = (mok, mnote)
    if breakout:
        res = float(breakout["res"])
        out["support"] = (True, f"المقاومة المخترقة {rnd(res)} صارت دعماً")
        out["retest"] = (bool(breakout.get("retest_ok")),
                         f"رجع لاختبار {rnd(res)} وثبت فوقه" if breakout.get("retest_ok")
                         else f"ما رجع لاختبار {rnd(res)} بعد أو ما ثبت فوقه")
        out["volume"] = (bool(breakout.get("vol_ok")),
                         "الاختراق كان بفوليوم أعلى من المعتاد" if breakout.get("vol_ok") else "الاختراق بدون فوليوم (ص19)")
        ok_c = bool(breakout.get("retest_ok"))
        out["candle"] = (ok_c, "شمعة ثبات فوق المستوى" if ok_c else "ما فيه شمعة ثبات على إعادة الاختبار")
    else:
        out["support"] = (len(eps) >= 1, f"منطقة {rnd(z_low)}–{rnd(z_high)}" + (f" لمسها {len(eps)} مرات" if eps else ""))
        ok5, note5 = False, "ما صار اختبار ثاني للدعم بعد القاع الأول"
        if len(eps) >= 2:
            a, b = eps[-1]
            if broke:
                note5 = f"كسر الدعم {rnd(z_low)} وبقي تحته (ص17: ليس سحب سيولة بل ضعف)"
            elif c[-1] < z_low:
                note5 = f"آخر إغلاق تحت الدعم {rnd(z_low)}"
            else:
                ok5 = True
                swept = float(np.min(l[a:b + 1])) < z_low * 0.995
                note5 = ("سحب سيولة تحت الدعم ثم استرداده" if swept else "رجع للدعم وما انكسر") + \
                        f" (الاختبار {len(eps)})"
        elif broke:
            note5 = f"كسر الدعم {rnd(z_low)} وبقي تحته (ص17: ليس سحب سيولة بل ضعف)"
        out["retest"] = (ok5, note5)
        red = [i for i in range(max(0, n - 11), n - 1) if c[i] < o[i]]
        ref = float(np.mean(v[red])) if red else float(np.mean(v[max(0, n - 21):n - 1]))
        out["volume"] = (bool(c[-1] >= o[-1] and v[-1] >= ref),
                         ("آخر شمعة حمراء — " if c[-1] < o[-1] else "")
                         + f"فوليومها {v[-1] / 1e3:.0f}K مقابل متوسط شموع البيع {ref / 1e3:.0f}K")
        info = candle(o[-1], h[-1], l[-1], c[-1], o[-2], c[-2])
        at = l[-1] <= zh and c[-1] >= z_low
        ok_c, nm = reversal(info, o[-1], h[-1], l[-1], c[-1], z_low, z_high)
        out["candle"] = (ok_c, (nm + " عند الدعم") if ok_c else
                         ("آخر شمعة يومية مو انعكاس" if at else
                          "آخر شمعة يومية أغلقت تحت الدعم" if l[-1] <= zh else "آخر شمعة بعيدة عن منطقة الدعم"))
    return {"checks": out, "critical": crit, "complete": all(x[0] for x in out.values()),
            "bar_date": str(df.index[-1])[:10], "frame": "يومي", "below_ma": below, "broke": broke,
            "episodes": len(eps), "breakout": bool(breakout)}


# ------------------------------------------------------------------ شمعة 4 ساعات (ص90)
def bars_4h(bars15: pd.DataFrame, day) -> pd.DataFrame:
    """شموع 4 ساعات للجلسة الرسمية من شموع 15 دقيقة: 9:30–13:30 و13:30–16:00 (نفس تقسيم منصات الشارت)."""
    if bars15 is None or not len(bars15):
        return pd.DataFrame()
    t = bars15[[ts.date() == day and REGULAR[0] <= ts.time() < REGULAR[1] for ts in bars15.index]]
    rows, idx = [], []
    for a, b in ((REGULAR[0], H4_SPLIT), (H4_SPLIT, REGULAR[1])):
        s = t[[a <= ts.time() < b for ts in t.index]]
        if len(s):
            rows.append([float(s["open"].iloc[0]), float(s["high"].max()), float(s["low"].min()),
                         float(s["close"].iloc[-1]), float(s["volume"].sum()), len(s)])
            idx.append(pd.Timestamp(datetime.combine(day, a), tz=t.index.tz))
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close", "volume", "n15"])


def p86_4h(daily_res: dict, bars15, z_low, z_high, prev_close, avg_vol, now_et, delay_min=0) -> dict:
    """نفس شروط ص86، والشمعة/الاختبار/الفوليوم من آخر شمعة 4 ساعات **مكتملة** اليوم (ص90 «اقربها 4 ساعات»).
    المتوسطات وRSI وMACD والدعم من اليومي المكتمل (الفريم الأكبر)."""
    if not daily_res or bars15 is None:
        return None
    b4 = bars_4h(bars15, now_et.date())
    if not len(b4):
        return None
    cutoff = now_et - timedelta(minutes=delay_min)
    ends = [datetime.combine(now_et.date(), H4_SPLIT, tzinfo=now_et.tzinfo),
            datetime.combine(now_et.date(), REGULAR[1], tzinfo=now_et.tzinfo)]
    done = [i for i in range(len(b4)) if ends[i] <= cutoff and b4["n15"].iloc[i] >= 8]
    if not done:
        return None
    i = done[-1]
    bar = b4.iloc[i]
    po, pc = (b4["open"].iloc[i - 1], b4["close"].iloc[i - 1]) if i > 0 else (prev_close, prev_close)
    info = candle(bar["open"], bar["high"], bar["low"], bar["close"], po, pc)
    zh = z_high * 1.03
    at = bar["low"] <= zh and bar["close"] >= z_low
    chk = dict(daily_res["checks"])
    low_today = float(b4["low"].iloc[:i + 1].min())
    crit = min(daily_res["critical"], low_today)
    if daily_res.get("breakout"):
        return None  # نموذج الاختراق: إعادة الاختبار تُقرأ على اليومي (ص68)
    if not chk["retest"][0] and at and not daily_res.get("broke") and daily_res.get("episodes", 0) >= 1:
        chk["retest"] = (True, ("سحب سيولة اليوم تحت الدعم ثم استرداده" if low_today < z_low * 0.995
                                else "رجع اليوم للدعم وما انكسر") + " (4 ساعات)")
    mins = (ends[i] - datetime.combine(now_et.date(), REGULAR[0] if i == 0 else H4_SPLIT,
                                       tzinfo=now_et.tzinfo)).total_seconds() / 60
    pace = avg_vol * mins / 390.0
    chk["volume"] = (bool(bar["close"] >= bar["open"] and bar["volume"] >= pace),
                     f"فوليوم شمعة 4 ساعات {bar['volume'] / 1e3:.0f}K مقابل المعتاد لنفس المدة {pace / 1e3:.0f}K")
    ok_c, nm = reversal(info, bar["open"], bar["high"], bar["low"], bar["close"], z_low, z_high)
    chk["candle"] = (ok_c, (nm + " على فريم 4 ساعات عند الدعم") if ok_c else
                     "آخر شمعة 4 ساعات مو انعكاس عند الدعم")
    chk["ma"] = (bool(daily_res.get("below_ma") and bar["close"] >= z_low and not daily_res.get("broke")),
                 ("أغلق تحت المتوسطات خلال القاعدة" if daily_res.get("below_ma") else "ما كان تحت المتوسطات الثلاثة")
                 + (f"، وشمعة 4 ساعات أغلقت فوق الدعم {rnd(z_low)}" if bar["close"] >= z_low else "، وتحت الدعم"))
    return {"checks": chk, "critical": float(crit), "complete": all(x[0] for x in chk.values()),
            "bar_date": f"{now_et.date()} {b4.index[i].strftime('%H:%M')}", "frame": "4 ساعات"}


# ------------------------------------------------------------------ الدخول والخروج
def ladder(z_low: float, z_high: float, critical: float) -> dict:
    """ص15–16 («قسم أوامرك: طلب 1، طلب 2، طلب 3 داخل منطقة التجميع»)، ص57 («كسرها خروج»):
    ثلاث طلبات Limit داخل المنطقة، والوقف تحت الدعم الحرج بتك واحد."""
    lo, hi = float(min(z_low, z_high)), float(max(z_low, z_high))
    if hi <= lo:
        hi = lo
    tk = tick(lo)
    top = float(np.floor(hi / tk + 1e-9) * tk)       # أعلى طلب ما يتعدى أعلى المنطقة
    bot = float(np.ceil(lo / tk - 1e-9) * tk)        # أدنى طلب ما ينزل تحت أسفل المنطقة
    if bot > top:
        bot = top
    mid = float(np.round(((top + bot) / 2) / tk) * tk)
    orders = [rnd(top), rnd(mid), rnd(bot)]
    stop = rnd(critical - tick(critical))
    avg = float(np.mean(orders))
    return {"orders": orders, "stop": stop, "critical": rnd(critical), "avg": avg,
            "risk": (avg - stop) / avg if avg > 0 else None, "risk_top": (orders[0] - stop) / orders[0],
            "risk_low": (orders[2] - stop) / orders[2]}


def reversal(info: dict, o, h, l, c, z_low, z_high) -> tuple:
    """شمعة الانعكاس عند الدعم (ص66: «Hammer / Engulfing / ذيل سفلي / رفض كسر»، ص82، ص86 «شموع رفض/ذيول سفلية»)."""
    at = l <= z_high * 1.03 and c >= z_low
    if not at:
        return False, ""
    if info["bullish_reversal"]:
        return True, candle_name(info)
    rng = max(h - l, 1e-9)
    if l < z_low * 0.998 and c > o and (c - l) / rng >= 0.6:
        return True, "رفض كسر (نزل تحت الدعم وأغلق فوقه)"
    return False, ""


def summary(checks: dict) -> tuple:
    met = [k for k in ORDER if k in checks and checks[k][0]]
    miss = [k for k in ORDER if k in checks and not checks[k][0]]
    return met, miss


def checks_text(checks: dict, short=False) -> str:
    out = []
    for k in ORDER:
        if k in checks:
            ok, note = checks[k]
            out.append(("✅ " if ok else "❌ ") + NAME[k] + ("" if short else f" ({note})"))
    return "، ".join(out)


def when_ready(setup: str, z_low, z_high, critical, res=None, t1=None) -> str:
    """ص70 «متى يكون جاهزاً؟» — بنفس طريقة جدول فيصل: مستوى محدد مسبقاً + ثبات."""
    zl, zh, cr = rnd(z_low), rnd(z_high), rnd(critical)
    if setup in ("BASE_BREAK", "BASE_RETEST", "W_BREAK", "W_RETEST") and res:
        return (f"رجوع لاختبار {rnd(res)} والثبات فوقه بشمعة يومية أو 4 ساعات (لا شراء للاختراق نفسه ص20، ص68)، "
                f"والطلبات Limit عند {rnd(res * 1.01)}–{rnd(res * 0.98)}")
    if setup == "W_FORMING" and res:
        return f"قاع ثاني يثبت عند {zl}–{zh} بشمعة انعكاس، أو كسر خط العنق {rnd(res)} والثبات ثم إعادة اختبار (ص25)"
    deep = rnd(z_low * 0.8), rnd(z_low * 0.9)
    return (f"اختبار منطقة {zl}–{zh} وفشل كسرها (أو سحب سيولة تحت {cr} ثم استرداد) + شمعة انعكاس يومية أو 4 ساعات "
            f"بفوليوم يتحسن (ص86)؛ وإذا كسرها وبقي تحتها فالانتظار أعمق 10–20% ({deep[0]}–{deep[1]}، ص54)")
