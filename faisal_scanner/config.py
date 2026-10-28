"""
إعدادات ماسح منهج فيصل — كل رقم هنا مأخوذ من الدليل (منهج فيصل.pdf).
Faisal-method scanner settings. Every threshold below is traced to the PDF guide.
"""
import os

# ---------------------------------------------------------------------------
# 1) فلتر الارتكاز — الأرقام الحرفية في method.py (ناسداك، تحت 5$، تحت 300M، فلووت تحت 20M، فوليوم فوق 100K)
#    27 أكتوبر: حُذفت كل الأرقام اللي ما هي من الملف (1.5× المخاطرة، 4 من 6، وقف 12%، Former Runner +30%، …)
# ---------------------------------------------------------------------------
EXCHANGES = os.getenv("EXCHANGES", "nasdaq,nyse,amex").split(",")  # نجلب الكل للعدّ، والفحص على ناسداك فقط (method.EXCHANGE)
MIN_PRICE = float(os.getenv("MIN_PRICE", "0.10"))      # اجتهاد: تجنب أسهم السنت (سبريد وتكات كبيرة)
MAX_PRICE = float(os.getenv("MAX_PRICE", "5.0"))       # ص6 Price Under $5
MAX_MARKET_CAP = float(os.getenv("MAX_MARKET_CAP", "300e6"))  # ص6 Under 300M
HOT_FLOAT = 5e6                                         # ص8 «أقل من 5M قوي وخطر»

# ---------------------------------------------------------------------------
# 2) المؤشرات (ص64، ص65، ص85)
# ---------------------------------------------------------------------------
MA_PERIODS = (20, 30, 50)
RSI_PERIOD = 14
RSI_IDEAL_LOW, RSI_IDEAL_HIGH = 23, 27   # ص65
RSI_LOOKBACK_BOUNCE = 15                 # للعرض في analyze.py
RSI_LOOKBACK_BREAKOUT = 30
PUMP_LOOKBACK = 22                       # اجتهاد: شمعة ضخ/تصريف خلال آخر شهر (ص10، ص20 قروبات الوهم)
MIN_DISTINCT_CLOSES = 8                  # اجتهاد: أسعار متقطعة تجعل قراءة الشموع بلا معنى
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

# ---------------------------------------------------------------------------
# 3) رسم الدعوم والمقاومات (اجتهاد في طريقة الرسم — ص13، ص17، ص21، ص25)
# ---------------------------------------------------------------------------
LOOKBACK_LEVELS = 120          # عدد الشموع اليومية لرسم الدعوم والمقاومات
SWING_K = 3                    # قمة/قاع محلي = أعلى/أدنى من 3 شموع يمين ويسار
ZONE_TOL = 0.04                # تجميع القيعان المتقاربة (4%) في منطقة طلب واحدة
NEAR_SUPPORT = 0.08            # «قريب من الدعم»
MAX_SWEEP_DEPTH = 0.10         # سحب السيولة = كسر بسيط تحت الدعم
W_MIN_NECK_HEIGHT = 0.08       # ارتفاع خط العنق فوق القاعين
W_MAX_LOW_DIFF = 0.06          # القاع الثاني قريب من الأول (أو أعلى منه)
DRAWDOWN_MED = 0.25            # نموذج الأسهم الهابطة (ص68): هبوط قبل القاعدة
INTRADAY_HOLD_BARS = 2         # ص25: ثبات فوق خط العنق «اكثر من ربع ساعه» (لمستويات فيصل المنشورة فقط)

# ---------------------------------------------------------------------------
# 4) فلتر الزخم — مُفعّل (ارتكاز + زخم)
# ---------------------------------------------------------------------------
# المستخدم (2026-10-19): «ابي أسهم الارتكاز فقط» → الزخم موقوف افتراضياً (لا قوائم ولا تنبيهات). ENABLE_MOMENTUM=1 يرجعه.
ENABLE_MOMENTUM = os.getenv("ENABLE_MOMENTUM", "0") == "1"
# --------------------------------------------------------------------------- (الفصل 3 + قاعدة المخاطرة الفصل 19)
# ---------------------------------------------------------------------------
MOM_MIN_CHANGE = 0.05          # Gap Up 5% / Change Up
MOM_MIN_VOLUME = 200_000       # Current Volume Over 200K
MOM_MIN_RVOL = 2.0             # Relative Volume Over 2
MOM_VWAP_ZONE = 0.02           # أول Pullback على الفواب: السعر ضمن 2% فوقه
MOM_EXTENDED = 0.15            # أبعد من 15% فوق الفواب = مطاردة
MOM_MAX_RISK = 0.02            # "في الزخم لا تسمح للخطأ أن يتجاوز 1% إلى 2%"
MOM_MIN_PULLBACK = 0.05        # لازم يكون فيه تراجع من القمة (مو شراء القمة)
# البري ماركت والأفتر ماركت
MOM_EXT_MIN_VOLUME = 100_000   # فوليوم الجلسة الممتدة
MOM_EXT_MIN_RVOL = 0.25        # فوليوم البري/الأفتر ≥ 25% من فوليوم يوم عادي = نشاط غير طبيعي
MOM_EXT_MAX_CANDIDATES = 40
MOM_QUICK_MAX_CANDIDATES = 80  # فحص الـ5 دقائق: أعلى 80 سهم صاعد (40 كانت تقطع أسهم +10%)

# ---------------------------------------------------------------------------
# 5) تحذيرات السلوك الخطر (الفصول 5، 10، 27–29)
# ---------------------------------------------------------------------------
PUMP_CHANGE = 0.40             # اجتهاد: شمعة قروبات وهم +40% وفوليوم 5x وذيل علوي طويل (ص10، ص20)
PUMP_VOL_MULT = 5.0
OFFERING_WORDS = (
    "offering", "registered direct", "private placement", "pricing of", "prices $",
    "warrant", "at-the-market", "atm program", "shelf", "s-1", "f-1", "dilution",
    "securities purchase agreement",
)
# «الخبر الإيجابي» (ص84 «يوجد خبر إيجابي أو فجوة علوية»، ص66 «محفز»): كلمات محفز واضحة بنفسها
CATALYST_WORDS = (
    "contract", "patent", "acquisition", "acquire", "merger", "approval", "approves", "clearance", "fast track",
    "orphan drug", "breakthrough", "partnership", "collaboration", "license agreement", "award", "buyback",
    "repurchase", "insider purchase", "insider buying", "purchase of shares by", "regains compliance",
)
# نتائج/إيرادات = إيجابية فقط إذا معها كلمة تحسن (النتائج الخاسرة مو محفز)
RESULT_WORDS = ("results", "earnings", "revenue", "quarter", "fiscal")
UP_WORDS = ("record", "increase", "grow", "growth", "beat", "exceed", "profit", "raise", "higher", "strong", "up ")
# ص72 «يخرج إذا بطل السبب: … خبر سلبي»، ص67/ص89 «خبر سلبي يلغي النموذج غالباً» — يلغي الإشارة
NEGATIVE_WORDS = (
    "bankruptcy", "chapter 11", "going concern", "delisting determination", "to be delisted", "will be delisted",
    "suspended from trading", "complete response letter", "rejects", "rejected", "discontinue", "default",
    "fails to meet primary", "did not meet", "net loss widen",
)
# تنبيه فقط (شائع في الأسهم تحت الدولار ويسبق التقسيم العكسي اللي فيصل يعتبره عامل اختيار ص53)
NOTICE_WORDS = ("deficiency", "non-compliance", "noncompliance", "minimum bid", "notice from nasdaq")
NEWS_DAYS = 7          # السلبي/الطرح: آخر أسبوع
CATALYST_DAYS = 3      # المحفز الإيجابي: آخر 3 أيام

# ---------------------------------------------------------------------------
# 6) مخرجات
# ---------------------------------------------------------------------------
MAX_WATCH_IN_REPORT = 15
MAX_MOMENTUM_IN_REPORT = 8
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")

# Alpaca: المفتاحان يوضعان كأسرار في GitHub (لا تكتبهما في الكود)
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
ALPACA_FEED = os.getenv("ALPACA_FEED", "sip")       # sip (متأخر 15 دقيقة في الخطة المجانية) أو iex
ALPACA_SIP_DELAY_MIN = int(os.getenv("ALPACA_SIP_DELAY_MIN", "16"))
