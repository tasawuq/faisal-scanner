"""
إعدادات ماسح منهج فيصل — كل رقم هنا مأخوذ من الدليل (منهج فيصل.pdf).
Faisal-method scanner settings. Every threshold below is traced to the PDF guide.
"""
import os

# ---------------------------------------------------------------------------
# 1) فلتر الارتكاز (الفصل 3 – Finviz)  |  Pivot universe filter
# ---------------------------------------------------------------------------
EXCHANGES = os.getenv("EXCHANGES", "nasdaq,nyse,amex").split(",")
MIN_PRICE = float(os.getenv("MIN_PRICE", "0.10"))      # تجنب أسهم السنت الصغيرة جداً
MAX_PRICE = float(os.getenv("MAX_PRICE", "5.0"))       # Price Under $5
MAX_MARKET_CAP = float(os.getenv("MAX_MARKET_CAP", "300e6"))  # Under 300M / Micro
MAX_FLOAT = float(os.getenv("MAX_FLOAT", "50e6"))      # Float Under 50M
PREFERRED_FLOAT = 20e6                                  # Under 20M مقبول للمضاربة
HOT_FLOAT = 5e6                                         # أقل من 5M قوي وخطر
MIN_AVG_VOLUME = 100_000                                # Average Volume Over 100K (فلتر الملف ص6) — شرط للتنبيه
WATCH_MIN_AVG_VOLUME = 50_000                           # 50K–100K: مراقبة فقط بدون تنبيه (مثال OMH 2026-10-20)

# ---------------------------------------------------------------------------
# 2) المتوسطات و RSI و MACD (الفصول 21–23)
# ---------------------------------------------------------------------------
MA_PERIODS = (20, 30, 50)
RSI_PERIOD = 14
RSI_IDEAL_LOW, RSI_IDEAL_HIGH = 23, 27   # قاعدة RSI 23–27
RSI_OVERSOLD = 30
RSI_LOOKBACK_BOUNCE = 15                 # منطقة الضغط يجب أن تكون حديثة (آخر 3 أسابيع تداول)
RSI_READY_MAX = 28                       # وضع صارم: لازم RSI يلمس 23–27 (أو أعمق) — 28 كحد أعلى
REVERSAL_MIN_ATR = 0.6                   # شمعة الانعكاس لازم تكون واضحة: مداها ≥ 0.6 من ATR
PUMP_LOOKBACK = 22                       # شمعة ضخ/تصريف خلال آخر شهر تمنع الجاهزية
MIN_DISTINCT_CLOSES = 8                  # أسعار متقطعة (تك سنت = 5%+) تجعل قراءة الشموع بلا معنى
RUNNER_LOOKBACK = 250                    # Former Runner: سبق انفجر خلال آخر سنة
RUNNER_CLOSE_JUMP = 0.30                 # إغلاق +30% في يوم
RUNNER_SPIKE = 0.50                      # أو قمة اليوم +50% فوق إغلاق أمس
RSI_LOOKBACK_BREAKOUT = 30               # في نموذج W/الاختراق القاع أقدم
RSI_LATE = 45                            # "RSI حول 40 يعني أن جزءاً من الارتداد حصل"
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

# ---------------------------------------------------------------------------
# 3) الدعم، سحب السيولة، W، الاختراق (الفصول 7، 9، 13، 25)
# ---------------------------------------------------------------------------
LOOKBACK_LEVELS = 120          # عدد الشموع اليومية لرسم الدعوم والمقاومات
SWING_K = 3                    # قمة/قاع محلي = أعلى/أدنى من 3 شموع يمين ويسار
ZONE_TOL = 0.04                # تجميع القيعان المتقاربة (4%) في منطقة طلب واحدة
NEAR_SUPPORT = 0.08            # "قريب من الدعم": أدنى سعر حديث ضمن 8% فوق المنطقة
MAX_SWEEP_DEPTH = 0.10         # سحب السيولة = كسر بسيط (حتى 10%) ثم استرداد
STOP_BUFFER = 0.02             # الوقف تحت القاع/الدعم بهامش 2%
MAX_RISK_PIVOT = 0.12          # في الارتكاز قد يكون الوقف أوسع لكن محدد
MIN_RR_T1 = 1.0                # "الدخول تحت مقاومة مباشرة" خطأ: أول هدف ≥ 1× المخاطرة
MIN_RR_SPACE = 2.5             # "خسارته قريبة وربحه كبير": أفضل هدف من أول 3 مقاومات ≥ 2.5× المخاطرة
RSI_LATE_BREAKOUT = 70         # بعد كسر خط العنق يرتفع RSI طبيعياً
CHASE_GAIN = 0.20              # الدخول الغلط: مطاردة بعد ارتفاع 20–50%
MAX_ABOVE_ENTRY = 0.05         # جاهز فقط إذا كان السعر ضمن 5% فوق منطقة الطلب
BIG_RUN = 0.50                 # حقق 50%+ → ينتظر Retest أو سحب سيولة
MAX_EXTENSION_BREAKOUT = 0.06  # بعد كسر خط العنق: لا دخول إذا ابتعد أكثر من 6%
W_MIN_NECK_HEIGHT = 0.08       # ارتفاع خط العنق فوق القاعين
W_MAX_LOW_DIFF = 0.06          # القاع الثاني قريب من الأول (أو أعلى منه)
DRAWDOWN_STRONG = 0.40         # ضغط بيع قوي = هبوط 40%+ من القمة القريبة
DRAWDOWN_MED = 0.25
SHORT_STRONG = 0.15            # شورت 15%+ من الفلووت
SHORT_MED = 0.08
MIN_SCORE_READY = 4.5          # من 6 شروط أساسية — "لا يكفي شرط واحد، اجمع الشروط معاً"
INTRADAY_HOLD_BARS = 2         # ثبات = آخر شمعتين 15 دقيقة فوق المستوى (أكثر من ربع ساعة)

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
PUMP_CHANGE = 0.40             # شمعة قروبات وهم: +40% وفوليوم 5x وذيل علوي طويل
PUMP_VOL_MULT = 5.0
REVERSE_SPLIT_DAYS = 120
OFFERING_WORDS = (
    "offering", "registered direct", "private placement", "pricing of", "prices $",
    "warrant", "at-the-market", "atm program", "shelf", "s-1", "f-1", "dilution",
    "securities purchase agreement",
)
CATALYST_WORDS = (
    "contract", "patent", "acquisition", "merger", "approval", "fda", "partnership",
    "agreement", "award", "results", "earnings", "revenue", "buyback", "insider",
)

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
