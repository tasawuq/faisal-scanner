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
MIN_FLOAT = 541e3            # أصغر سهم في أمثلته: HTCR فلوت 541 ألف (ص90) — أقل من كذا خارج مدى أسهمه
MIN_AVG_VOLUME = 100_000     # ص6، ص61 Average Volume Over 100K
RUNNER_MIN = 1.00            # ص10 «Former Runner: سبق انفجر 100% أو أكثر»
RUNNER_WINDOW = 21           # اجتهاد: «خلال شهر» — LFS ص32 «صعد معنا قبل اقل من شهر من 1.60 ل 5»
JUST_DOUBLED = 1.00          # ص59 «توه مدبل فيه خطوره» / «توه صاعد من سنتات ل 3 غير امن الان»
JUST_DOUBLED_BARS = 10       # اجتهاد: «توه» = آخر أسبوعين تداول
RECENT_SPLIT_DAYS = 210      # أمثلة الملف: تقسيمات أسهم فيصل خلال أيام إلى ~7 أشهر (ص56، ص59، ص90–93)
# دليل فيصل التدريجي (الخطوة 7): «الـRSI يقيس التشبع البيعي على الفريم اليومي: ما دون 30 = تشبع بيعي». وص86 «RSI في
# التشبع البيعي 23 - 27» = أمثلة داخل التشبع البيعي (2 نوفمبر: كان الكود 27 — أضيق من تعريفه)
RSI_PRESSURE = 30
CHASE_GAIN = 0.20            # ص67، ص88 «مطاردة السهم بعد ارتفاع 20-50%»
MAX_LOSS = 0.15              # ص54 «نسبة الخطا الاسبوعيه الاصل لاتتجاوز خساره 15٪»
MIN_BASE_DAYS = 5            # الدليل خ7 «أدنى شمعة هبط لها السهم خلال 5 جلسات: هل كُسرت على جميع الفريمات؟» (+ ص82 «قاعدة سعرية»)
BASE_WINDOW = 40             # اجتهاد: نافذة القاعدة على اليومي (~شهرين) لقراءة القاع وإعادة الاختبار وRSI
REGULAR = (time(9, 30), time(16, 0))
# الدليل خ5: «الفريم اليومي: شموع أوقات الماركت فقط — فريم 4 ساعات: جميع أوقات التداول» (البري والماركت والأفتر):
# 4:00–8:00، 8:00–12:00، 12:00–16:00، 16:00–20:00 نيويورك (نفس تقسيم الشارت مع الجلسات الممتدة)
H4_BARS = ((time(4, 0), time(8, 0)), (time(8, 0), time(12, 0)), (time(12, 0), time(16, 0)), (time(16, 0), time(20, 0)))


PAGE = {"ma": "ص66/ص82/ص84/ص86", "target": "ص66/ص84/ص88", "rsi": "ص65/ص86/الدليل خ7", "macd": "ص85/ص86", "support": "ص86", "retest": "ص86/ص17",
        "volume": "ص85/ص86", "candle": "ص82/ص86", "pressure": "ص85", "red": "ص88", "accum": "ص12/ص13",
        "groups": "الدليل خ7"}
NAME = {"ma": "تحت المتوسطات ثم بدأ يستعيدها", "target": "محفز أو هدف علوي", "rsi": "RSI نزل تحت 30 (تشبع بيعي)", "macd": "MACD يتحسن أو تقاطع",
        "support": "دعم/قاع واضح", "retest": "اختبار الدعم وفشل الكسر", "volume": "فوليوم يتحسن مع الارتداد",
        "candle": "شمعة انعكاس عند الدعم", "pressure": "ضغط بيع سابق ثم ضعف الهبوط",
        "red": "ما فيه شموع حمراء طويلة قرب القاع", "accum": "تجميع: قيعان أعلى وشموع تتقلص",
        "groups": "خالٍ من القروبات: تذبذب ≤10%"}
ORDER = ("ma", "rsi", "macd", "support", "retest", "accum", "volume", "candle", "pressure", "red", "groups", "target")
# الدليل خ7 «سهم خالٍ من القروبات: يحافظ على تذبذب نقاط 10% كحد أقصى (القروبات ترفع السهم بشموع طويلة بدون جسم تصل 20%
# على فريم 4 ساعات)» — على اليومي: مدى كل شمعة (أعلى−أدنى) ≤10% في الجلسات الخمس قبل شمعة الإشارة (اجتهاد في النافذة: نفس
# «خلال 5 جلسات» في البند اللي بعده؛ وشمعة الإشارة نفسها ما تنحسب — شمعة الانعكاس طبيعي تكون أكبر)
GROUP_SWING = 0.10
GROUP_BAR_4H = 0.20
GROUP_SESSIONS = 5
PRESSURE_DROP = 0.25         # اجتهاد: «هبط بقوة» (ص85) = 25%+ من أعلى سعر في 90 يوم


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
def p86_daily(df: pd.DataFrame, z_low: float, z_high: float, breakout: dict = None, kind: str = "support") -> dict:
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
    # زيارة الدعم = قاع الشمعة لمس المنطقة (هامش 1%)؛ الزيارات يفصلها يومان على الأقل فوقها (اختبار أول ثم ثاني)
    eps = [] if breakout else episodes(l, z_high * 1.01, st, n)
    broke = False
    first = n - 3  # الثبات = آخر 3 إغلاقات فوق الدعم الحرج
    if breakout:
        # الحرج تحت المنطقة دائماً (نفس الرقم اللي ينطبع للمستخدم)، والثبات يُحسب من شمعة الاختراق وبعدها فقط —
        # الإغلاقات قبل الاختراق طبيعي تكون تحت المقاومة (BDRX 27 أكتوبر: إغلاق 3.384 قبل الاختراق حُسب كسر بالغلط)
        crit = float(min(breakout.get("retest_low") or z_low, z_low))
        if breakout.get("brk") is not None:
            first = max(first, int(breakout["brk"]))
    else:
        last_ep = eps[-1] if eps else None
        crit = float(min(z_low, np.min(l[last_ep[0]:last_ep[1] + 1]))) if last_ep else float(z_low)
        if last_ep:
            # ص17: «إذا كسر الدعم وبقي تحته أو فشل في الرجوع فوقه → ليس سحب سيولة بل ضعف» (إغلاقان متتاليان تحته)
            broke = any(c[i] < z_low * 0.99 and c[i + 1] < z_low * 0.99 for i in range(last_ep[0], n - 1))
    holding = bool(np.all(c[first:] >= crit))
    # المتوسطات (فهم الصفحات مع بعض): تحتها = ضغط (ص85، ص86 «السعر تحت المتوسطات ثم يبدأ التماسك») ثم «يبدأ يستعيدها»
    # (ص66) / «بدأ يستعيد المتوسطات» (ص84) / «عودة السعر للإغلاق فوق متوسط 30 و50» (ص82) = تأكيد الخروج من الضعف.
    # البداية = إغلاق فوق أقرب متوسط (MA20)؛ والإغلاق فوق 30 و50 تأكيد أقوى (ص82). المتوسطات فلتر تأكيد (ص64):
    # الإشارة عند الاستعادة، والدخول بطلبات Limit راجعة للدعم (ص86 دخول 2: «بعد إعادة اختبار الدعم بنجاح»)
    m20, m30, m50 = float(ma[20][-1]), float(ma[30][-1]), float(ma[50][-1])
    reclaim = bool(np.isfinite(m20) and c[-1] > m20)
    strong = bool(np.isfinite(m30) and np.isfinite(m50) and c[-1] > m30 and c[-1] > m50)
    out["ma"] = (below and reclaim and holding,
                 ("كان تحت المتوسطات الثلاثة خلال القاعدة" if below else "ما كان تحت المتوسطات الثلاثة خلال القاعدة")
                 + (("، وأغلق فوق 30 و50 (ص82)" if strong else f"، وبدأ يستعيدها: أغلق فوق MA20 {rnd(m20)}")
                    if reclaim else f"، وما بدأ يستعيدها (MA20 {rnd(m20)} فوق السعر)")
                 + ("" if holding else f"، وأغلق تحت الدعم الحرج {rnd(crit)}"))
    rmin = float(np.nanmin(r[st:])) if n > st else float("nan")
    out["rsi"] = (bool(np.isfinite(rmin) and rmin < RSI_PRESSURE), f"أدنى RSI {rmin:.1f} (الحالي {r[-1]:.0f})")
    mok, mnote = macd_ok(line, sig, hist)
    out["macd"] = (mok, mnote)
    if breakout:
        res = float(breakout["res"])
        out["support"] = (True, f"المقاومة المخترقة {rnd(res)} صارت دعماً")
        out["retest"] = (bool(breakout.get("retest_ok")),
                         f"رجع لاختبار {rnd(res)} وثبت فوقه" if breakout.get("retest_ok")
                         else f"ما رجع لاختبار {rnd(res)} بعد أو ما ثبت فوقه")
        bar_ok = breakout.get("bar_ok", True)
        out["volume"] = (bool(breakout.get("vol_ok") and bar_ok),
                         "الاختراق بذيل علوي طويل وإغلاق ضعيف — علامة اختراق وهمي (ص19)" if not bar_ok else
                         "الاختراق كان بفوليوم أعلى من المعتاد" if breakout.get("vol_ok") else "الاختراق بدون فوليوم (ص19)")
        ok_c = bool(breakout.get("retest_ok"))
        out["candle"] = (ok_c, "شمعة ثبات فوق المستوى" if ok_c else "ما فيه شمعة ثبات على إعادة الاختبار")
    else:
        # «بدأ يكوّن قاعدة سعرية قرب دعم واضح» (ص82، ص66): الدعم عمره أسبوع على الأقل (اجتهاد في المدة — EDBL 22–25 سبتمبر:
        # «دعم» عمره 3 أيام من شمعة انهيار خبر ما هو قاعدة)
        age = (n - 1 - eps[0][0]) if eps else 0
        out["support"] = (len(eps) >= 1 and age >= MIN_BASE_DAYS,
                          f"منطقة {rnd(z_low)}–{rnd(z_high)}" + (f" لمسها {len(eps)} مرات خلال {age} يوم" if eps else "")
                          + ("" if age >= MIN_BASE_DAYS else " — ما تكوّنت قاعدة بعد (أقل من أسبوع)"))
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
        at = l[-1] <= zh and c[-1] >= z_low
        ok_c, nm = reversal_seq(o, h, l, c, z_low, z_high)
        out["candle"] = (ok_c, (nm + " عند الدعم") if ok_c else nm if nm else
                         ("آخر شمعة يومية مو انعكاس" if at else
                          "آخر شمعة يومية أغلقت تحت الدعم" if l[-1] <= zh else "آخر شمعة بعيدة عن منطقة الدعم"))
    # ص12–ص13 «كيف تقرأ تجميع المضارب»: «التجميع = دعم واضح + تماسك + فوليوم بيع أقل + قيعان أعلى»، و«الحظ تقلص
    # الشموع: الشموع تصبح أصغر والهبوط أقل حدة». ينطبق على الارتداد من الدعم فقط — سحب السيولة (قاع أدنى بطبيعته)
    # ونموذج W والاختراق أنواع سلوك ثانية لها دخولها (ص9 «حدد نوع السلوك أولاً، ثم اختر طريقة الدخول المناسبة له»)
    r_ = (h - l) / np.maximum(c, 1e-9)
    rec_, pri_ = r_[max(0, n - 6):n - 1], r_[max(0, n - 16):max(0, n - 6)]
    shrink = bool(len(rec_) and len(pri_) and np.mean(rec_) < np.mean(pri_))  # اجتهاد: آخر 5 شموع مقابل الـ10 قبلها
    ep_lows = [float(np.min(l[a:b + 1])) for a, b in eps]
    if breakout or kind != "support":
        why = "اختراق قاعدة" if breakout else {"sweep": "سحب سيولة", "w": "نموذج W"}.get(kind, kind)
        out["accum"] = (True, f"لا ينطبق — {why} نوع سلوك ثاني غير التجميع (ص9)")
    else:
        hl = len(ep_lows) >= 2 and ep_lows[-1] > ep_lows[-2]
        out["accum"] = (bool(hl and shrink),
                        ((f"قاع الاختبار الأخير {rnd(ep_lows[-1])} أعلى من اللي قبله {rnd(ep_lows[-2])}" if hl else
                          f"قاع الاختبار الأخير {rnd(ep_lows[-1])} مو أعلى من اللي قبله {rnd(ep_lows[-2])}")
                         if len(ep_lows) >= 2 else "ما فيه قاعين للمقارنة بعد")
                        + ("، والشموع تتقلص" if shrink else "، والشموع ما تقلصت"))
    # ص85 الشرط 4: «وجود شورت أو ضغط بيع سابق ثم ضعف الهبوط» — هبوط قوي ثم ضعف البيع (فوليوم شموع البيع يقل أو الدعم صمد)
    hi90 = float(np.nanmax(h[max(0, n - 90):n]))
    drop = 1 - c[-1] / hi90 if hi90 > 0 else 0.0
    red_i = [i for i in range(max(0, n - 20), n) if c[i] < o[i]]
    rv5 = [v[i] for i in red_i if i >= n - 5]
    rv15 = [v[i] for i in red_i if i < n - 5]
    drying = bool(rv5 and rv15 and np.mean(rv5) < 0.8 * np.mean(rv15)) or bool(rv15 and not rv5)
    weak = drying or bool(out.get("retest", (False,))[0])
    out["pressure"] = (bool(drop >= PRESSURE_DROP and weak),
                       f"هبط {drop * 100:.0f}% من أعلى 90 يوم" + ("، والبيع ضعف" if weak else "، والبيع ما ضعف بعد"))
    # (RSI «حول 40» في ص65/ص78/ص80 = RSI لما يبدأ النموذج «40+ من البداية» — يغطيه شرط «RSI وصل 23–27» في القاعدة.
    #  والدخول نفسه بطلبات Limit راجعة للدعم (ص86 دخول 2)، فما فيه شراء والـRSI مرتفع بعد الارتداد)
    # ص88 «علامات سلبية: شموع حمراء طويلة قرب القاع»: آخر 3 شموع، جسم أحمر ≥ متوسط مدى 14 يوم وقريبة من المنطقة
    rngs = h[max(0, n - 15):n - 1] - l[max(0, n - 15):n - 1]
    atr = float(np.mean(rngs)) if len(rngs) else 0.0
    bad = [i for i in range(max(0, n - 3), n) if c[i] < o[i] and (o[i] - c[i]) >= atr and l[i] <= zh]
    out["red"] = (not bad, "ما فيه" if not bad else f"شمعة حمراء طويلة {str(df.index[bad[-1]])[:10]} عند القاع")
    sw = (h - l) / np.maximum(l, 1e-9)
    win = sw[max(0, n - 1 - GROUP_SESSIONS):n - 1]
    mx = float(np.max(win)) if len(win) else 0.0
    out["groups"] = (bool(mx <= GROUP_SWING + 1e-9),
                     f"أكبر تذبذب يومي آخر {GROUP_SESSIONS} جلسات {mx * 100:.1f}%"
                     + ("" if mx <= GROUP_SWING + 1e-9 else " — فوق 10%: سلوك قروبات (الدليل خ7)"))
    return {"checks": out, "critical": crit, "complete": all(x[0] for x in out.values()),
            "bar_date": str(df.index[-1])[:10], "frame": "يومي", "below_ma": below, "broke": broke,
            "ma20": float(ma[20][-1]),
            "episodes": len(eps), "breakout": bool(breakout), "kind": "breakout" if breakout else kind,
            "ep_lows": ep_lows, "ep_open": bool(eps and eps[-1][1] >= n - 1), "shrink": shrink}


# ------------------------------------------------------------------ شمعة 4 ساعات (ص90)
def bars_4h(bars15: pd.DataFrame, after=None) -> pd.DataFrame:
    """شموع 4 ساعات لكل أوقات التداول (الدخول خ5) من شموع 15 دقيقة: 4–8، 8–12، 12–16، 16–20 نيويورك.
    after = تاريخ آخر شمعة يومية مكتملة: نأخذ فقط شموع 4 ساعات اللي بعد إغلاقها (أفتر ذاك اليوم وما بعده)."""
    if bars15 is None or not len(bars15):
        return pd.DataFrame()
    rows, idx = [], []
    for day in sorted({ts.date() for ts in bars15.index}):
        if after is not None and day < after:
            continue
        t = bars15[[ts.date() == day for ts in bars15.index]]
        for a, b in H4_BARS:
            if after is not None and day == after and a < REGULAR[1]:
                continue  # ماركت يوم الشمعة اليومية المكتملة = داخل الشمعة اليومية نفسها
            s = t[[a <= ts.time() < b for ts in t.index]]
            if len(s):
                reg = s[[REGULAR[0] <= ts.time() < REGULAR[1] for ts in s.index]]
                rows.append([float(s["open"].iloc[0]), float(s["high"].max()), float(s["low"].min()),
                             float(s["close"].iloc[-1]), float(s["volume"].sum()), len(s), len(reg)])
                idx.append(pd.Timestamp(datetime.combine(day, a), tz=s.index.tz))
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close", "volume", "n15", "n15_reg"])


def _h4_end(ts) -> datetime:
    return datetime.combine(ts.date(), dict(H4_BARS)[ts.time()], tzinfo=ts.tzinfo)


def _reg_minutes(ts) -> float:
    a = datetime.combine(ts.date(), ts.time(), tzinfo=ts.tzinfo)
    b = _h4_end(ts)
    lo = max(a, datetime.combine(ts.date(), REGULAR[0], tzinfo=ts.tzinfo))
    hi = min(b, datetime.combine(ts.date(), REGULAR[1], tzinfo=ts.tzinfo))
    return max(0.0, (hi - lo).total_seconds() / 60)


def p86_4h(daily_res: dict, bars15, z_low, z_high, prev_close, avg_vol, now_et, delay_min=0, after=None) -> dict:
    """نفس شروط ص86، والشمعة/الاختبار/الفوليوم من آخر شمعة 4 ساعات **مكتملة** بعد آخر شمعة يومية مكتملة (ص90 «اقربها
    4 ساعات»؛ الدليل خ5: 4 ساعات = جميع أوقات التداول، فتدخل شموع الأفتر والبري). المتوسطات وRSI وMACD والدعم من اليومي
    المكتمل (الفريم الأكبر). after = تاريخ آخر شمعة يومية مكتملة."""
    if not daily_res or bars15 is None:
        return None
    if after is None:
        after = now_et.date() - timedelta(days=1)
    b4 = bars_4h(bars15, after)
    if not len(b4):
        return None
    cutoff = now_et - timedelta(minutes=delay_min)
    # مكتملة = وقت نهايتها مضى (مع تأخير البيانات)، وشموع الماركت فيها ما تنقص (8 من 10 أو 16 شمعة 15 دقيقة)
    done = [i for i in range(len(b4)) if _h4_end(b4.index[i]) <= cutoff
            and b4["n15_reg"].iloc[i] >= 0.8 * _reg_minutes(b4.index[i]) / 15]
    if not done:
        return None
    i = done[-1]
    bar = b4.iloc[i]
    po, pc = (b4["open"].iloc[i - 1], b4["close"].iloc[i - 1]) if i > 0 else (prev_close, prev_close)
    zh = z_high * 1.03
    chk = dict(daily_res["checks"])
    low_today = float(b4["low"].iloc[:i + 1].min())  # أدنى سعر بعد آخر إغلاق يومي — كل الفريمات (الدليل خ7)
    crit = min(daily_res["critical"], low_today)
    if daily_res.get("breakout"):
        return None  # نموذج الاختراق: إعادة الاختبار تُقرأ على اليومي (ص68)
    # لمس المنطقة بعد آخر إغلاق يومي (بأي شمعة 4 ساعات — المطرقة ثم شمعة التأكيد) والإغلاق الحالي فوق الدعم
    touched = low_today <= zh and bar["close"] >= z_low
    if not chk["retest"][0] and touched and not daily_res.get("broke") and daily_res.get("episodes", 0) >= 1:
        chk["retest"] = (True, ("سحب سيولة تحت الدعم ثم استرداده" if low_today < z_low * 0.995
                                else "رجع للدعم وما انكسر") + " (4 ساعات)")
    if daily_res.get("kind") == "support" and "accum" in chk and not chk["accum"][0] and touched \
            and low_today < z_low * 0.995:
        # نزل تحت الدعم ثم استرده = سحب سيولة (ص16) — نوع سلوك ثاني، قاعه أدنى بطبيعته (ص9)
        chk["accum"] = (True, "لا ينطبق — سحب سيولة تحت الدعم ثم استرداده (ص9، ص16)")
    elif daily_res.get("kind") == "support" and "accum" in chk and not chk["accum"][0] and touched:
        lows = list(daily_res.get("ep_lows") or [])
        if lows and daily_res.get("ep_open"):
            lows[-1] = min(lows[-1], low_today)
        else:
            lows.append(low_today)
        hl = len(lows) >= 2 and lows[-1] > lows[-2]
        sh = bool(daily_res.get("shrink"))
        chk["accum"] = (bool(hl and sh), (f"القاع الجديد {rnd(lows[-1])} " + ("أعلى" if hl else "مو أعلى")
                                          + f" من الاختبار اللي قبله {rnd(lows[-2])}" if len(lows) >= 2 else
                                          "ما فيه قاعين للمقارنة بعد")
                        + ("، والشموع تتقلص" if sh else "، والشموع ما تقلصت") + " (4 ساعات)")
    # الفوليوم المعتاد لنفس المدة من متوسط اليوم؛ شمعة البري/الأفتر وحدها: كأنها ساعة ماركت (اجتهاد — فوليوم الجلسات
    # الممتدة أقل بطبيعته)
    pace = avg_vol * max(_reg_minutes(b4.index[i]), 60.0) / 390.0
    chk["volume"] = (bool(bar["close"] >= bar["open"] and bar["volume"] >= pace),
                     f"فوليوم شمعة 4 ساعات {bar['volume'] / 1e3:.0f}K مقابل المعتاد لنفس المدة {pace / 1e3:.0f}K")
    seq = b4.iloc[:i + 1]
    if len(seq) < 3:  # شموع 4 ساعات قبلها من آخر يوم مكتمل غير متوفرة → شمعة اليوم اليومية السابقة كأساس
        pre = pd.DataFrame([[po, max(po, pc), min(po, pc), pc]], columns=["open", "high", "low", "close"])
        seq = pd.concat([pre, seq[["open", "high", "low", "close"]]], ignore_index=True)
    so, sh_, sl, sc = (seq[k].values.astype(float) for k in ("open", "high", "low", "close"))
    ok_c, nm = reversal_seq(so, sh_, sl, sc, z_low, z_high)
    chk["candle"] = (ok_c, (nm + " على فريم 4 ساعات عند الدعم") if ok_c else
                     (nm + " (4 ساعات)") if nm else "آخر شمعة 4 ساعات مو انعكاس عند الدعم")
    # القروبات على 4 ساعات (الدليل خ7): شمعة طويلة بدون جسم تصل 20% بعد آخر إغلاق يومي
    rng4 = (b4["high"] - b4["low"]) / b4["low"].clip(lower=1e-9)
    body4 = (b4["close"] - b4["open"]).abs() / (b4["high"] - b4["low"]).clip(lower=1e-9)
    grp = [j for j in range(i + 1) if rng4.iloc[j] >= GROUP_BAR_4H and body4.iloc[j] <= 1 / 3]
    if grp and "groups" in chk:
        j = grp[-1]
        chk["groups"] = (False, f"شمعة 4 ساعات {b4.index[j].strftime('%H:%M')} مداها {rng4.iloc[j] * 100:.0f}% بدون جسم — "
                                "سلوك قروبات (الدليل خ7)")
    m20 = daily_res.get("ma20")
    rec = bool(m20 and bar["close"] > m20)
    chk["ma"] = (bool(daily_res.get("below_ma") and rec and bar["close"] >= z_low and not daily_res.get("broke")),
                 ("كان تحت المتوسطات خلال القاعدة" if daily_res.get("below_ma") else "ما كان تحت المتوسطات الثلاثة")
                 + (f"، وشمعة 4 ساعات أغلقت فوق MA20 {rnd(m20)}" if rec else f"، وما بدأ يستعيدها (MA20 {rnd(m20 or 0)})"))
    t0 = b4.index[i]
    seg = "بري" if t0.time() < REGULAR[0] else "أفتر" if t0.time() >= REGULAR[1] else "ماركت"
    return {"checks": chk, "critical": float(crit), "complete": all(x[0] for x in chk.values()),
            "bar_date": f"{t0.date()} {t0.strftime('%H:%M')}", "frame": "4 ساعات", "segment": seg,
            "low_since_close": low_today}


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


def reversal_seq(o, h, l, c, z_low, z_high) -> tuple:
    """شمعة الانعكاس على آخر شمعة مكتملة من سلسلة (يومي أو 4 ساعات) — مقاطع الشموع اللي أرسلها المستخدم (2 نوفمبر):
    - الابتلاعية الشرائية (شمعتين): الثانية خضراء تبتلع جسم الحمراء كامل → تأكيدها فيها.
    - نجمة الصباح (3 شموع): حمراء طويلة، ثم جسم صغير تحتها، ثم خضراء تدخل بعمق داخل الحمراء (فوق نصف جسمها).
    - المطرقة/الذيل السفلي/رفض الكسر (شمعة وحدة): «من الأهمية انتظار التأكيد في اليوم التالي» → الإشارة على الشمعة اللي
      بعدها إذا أغلقت خضراء فوق جسم المطرقة.
    يرجع (ok, الاسم)."""
    n = len(c)
    if n < 2:
        return False, ""
    info = candle(o[-1], h[-1], l[-1], c[-1], o[-2], c[-2])
    at = l[-1] <= z_high * 1.03 and c[-1] >= z_low
    if at and info["engulfing"]:
        return True, candle_name(info)
    if n >= 3 and c[-1] >= z_low:
        b1 = o[-3] - c[-3]
        small = abs(c[-2] - o[-2]) <= b1 / 3
        near = min(l[-3], l[-2], l[-1]) <= z_high * 1.03
        if b1 > 0 and b1 >= 0.5 * (h[-3] - l[-3]) and small and max(o[-2], c[-2]) <= c[-3] + b1 * 0.25 \
                and c[-1] > o[-1] and c[-1] >= c[-3] + b1 / 2 and near:
            return True, "نجمة الصباح (3 شموع)"
    # مطرقة/ذيل سفلي/رفض كسر أمس + تأكيد اليوم
    pinfo = candle(o[-2], h[-2], l[-2], c[-2], o[-3] if n >= 3 else None, c[-3] if n >= 3 else None)
    pok, pname = reversal(dict(pinfo, engulfing=False, bullish_reversal=pinfo["hammer"] or pinfo["lower_wick"]),
                          o[-2], h[-2], l[-2], c[-2], z_low, z_high)
    if pok and c[-1] > o[-1] and c[-1] > max(o[-2], c[-2]) and c[-1] >= z_low:
        return True, f"{pname} ثم شمعة تأكيد خضراء"
    sok, sname = reversal(dict(info, engulfing=False, bullish_reversal=info["hammer"] or info["lower_wick"]),
                          o[-1], h[-1], l[-1], c[-1], z_low, z_high)
    if sok:
        return False, f"{sname} — تنتظر شمعة تأكيد بعدها (مقطع المطرقة: «انتظار التأكيد في اليوم التالي»)"
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
