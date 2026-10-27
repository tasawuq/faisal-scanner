"""تقرير عربي مختصر يصلح للجوال."""
from datetime import datetime
from zoneinfo import ZoneInfo

from . import config as C

RIYADH = ZoneInfo("Asia/Riyadh")
SESSION_AR = {"pre": "قبل الافتتاح (بري ماركت)", "regular": "السوق مفتوح", "post": "بعد الإغلاق (أفتر ماركت)",
              "closed": "السوق مغلق"}


def p(x):
    if x is None:
        return "—"
    return f"{x:.2f}" if x >= 1 else f"{x:.3f}" if x >= 0.1 else f"{x:.4f}"


def big(x):
    if not x:
        return "—"
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def _float_txt(s):
    f = s.get("meta", {}).get("float")
    return f"فلووت {big(f)}" if f else "فلووت غير متوفر"


def ready_block(s: dict) -> str:
    lines = [f"🚨 **{s['ticker']}** — ${p(s['price'])} | {s['setup_ar']}"]
    if s["kind"] == "pivot":
        stop_pct = f" (−{s['risk_pct']*100:.1f}%)" if s.get("risk_pct") is not None else ""
        o = ((s.get("meta") or {}).get("ladder") or {}).get("orders") or [s['entry_high'], s['entry_low']]
        lines.append(f"• طلبات Limit داخل منطقة الطلب: {' / '.join(p(x) for x in o)} | "
                     f"الوقف: {p(s['stop'])}{stop_pct}")
        tg = f"{p(s['t1'])} ({s['t1_label']})" if s.get("t1") else "—"
        if s.get("t2"):
            tg += f" ← {p(s['t2'])} ({s['t2_label']})"
        rr = f" | عائد/مخاطرة {s['rr']:.1f}" if s.get("rr") else ""
        lines.append(f"• الأهداف: {tg}{rr}")
        bits = [f"RSI {s['rsi_now']:.0f} (أدنى {s['rsi_min']:.0f})", f"MACD {s['macd_state']}", s["ma_state"]]
        if s.get("candle"):
            bits.append(s["candle"])
        lines.append("• " + " · ".join(bits))
        extra = [_float_txt(s), f"شروط ص86 {s['score']:.0f}/7"]
    else:
        m = s["meta"]
        lines.append(f"• دخول قرب {p(s['entry_low'])} | الوقف {p(s['stop'])} (−{(s['risk_pct'] or 0)*100:.1f}%) | "
                     f"الهدف {p(s.get('t1'))}")
        extra = [f"{m['change']*100:+.0f}%", s.get("volume_note", ""), f"الفواب {p(m['vwap'])}",
                 f"بداية الحركة {p(m.get('move_base'))}", _float_txt(s)]
    extra += s.get("tags", [])
    lines.append("• " + " · ".join(extra))
    m = s.get("meta", {})
    if m.get("ext_price"):
        seg = "البري" if m.get("ext_session") == "pre" else "الأفتر"
        lines.append(f"• سعر {seg} الآن: {p(m['ext_price'])} ({(m.get('ext_chg') or 0)*100:+.1f}%)")
    if s.get("news"):
        lines.append(f"• خبر: {s['news'][0]['headline'][:110]}")
    if s.get("flags"):
        lines.append("• ⚠️ " + " · ".join(s["flags"]))
    return "\n".join(lines)


def watch_line(s: dict) -> str:
    zone = "منطقة الطلب" if s["setup"] in ("AT_SUPPORT", "SWEEP", "BOUNCE") else "الدخول بعد التأكيد"
    head = (f"• **{s['ticker']}** ${p(s['price'])} — {s['setup_ar']} | {zone} {p(s['entry_low'])}–{p(s['entry_high'])} "
            f"| الوقف {p(s['stop'])} | RSI {s['rsi_now']:.0f} (أدنى {s['rsi_min']:.0f}) | ص86 {s['score']:.0f}/7")
    w = "؛ ".join(s.get("waiting_for", [])[:2])
    out = head + (f"\n   ينتظر: {w}" if w else "")
    if s.get("flags"):
        out += "\n   ⚠️ " + " · ".join(s["flags"][:3])
    return out


def mom_line(s: dict) -> str:
    m = s["meta"]
    out = (f"• **{s['ticker']}** ${p(s['price'])} {m['change']*100:+.0f}% | الفواب {p(m['vwap'])} | "
           f"{s.get('volume_note', '')} | {_float_txt(s)} | {s['setup_ar']}")
    if s.get("waiting_for"):
        out += f"\n   ينتظر: {s['waiting_for'][0]}"
    if s.get("flags"):
        out += "\n   ⚠️ " + " · ".join(s["flags"][:3])
    return out


def build_markdown(result: dict) -> str:
    now_et = datetime.fromisoformat(result["generated_at_et"])
    now_ry = now_et.astimezone(RIYADH)
    L = [f"## 📋 منهج فيصل (ارتكاز + زخم) — {now_ry:%H:%M} الرياض ({now_et:%H:%M} نيويورك)",
         f"{SESSION_AR.get(result['session'], result['session'])} · فُحص {result['counts']['universe']} سهم أمريكي → "
         f"{result['counts']['analyzed']} يطابق فلتر فيصل (ناسداك، سعر < $5، قيمة < 300M، فوليوم > 100K، فلووت/عدد أسهم ≤ 8M مثل أسهم فيصل)", ""]
    ready = result["ready"]
    L.append(f"### 🚨 اكتملت شروط ص86 — طلبات Limit ({len(ready)})")
    if ready:
        for s in ready:
            L += [ready_block(s), ""]
    else:
        L += ["لا يوجد سهم اكتملت شروطه الآن — الصبر جزء من المنهج.", ""]
    if result.get("faisal"):
        L.append(f"### 📌 أسهم فيصل — مستوياته المنشورة ({len(result['faisal'])})")
        for f in result["faisal"]:
            line = f"• **{f['ticker']}** {f['price_label']} {p(f['price'])} — {f['state_ar']}"
            if f.get("note"):
                line += f"\n   فيصل: {f['note']}"
            sc = f.get("scanner")
            if sc:
                line += f"\n   شروط ص86: {'✅ اكتملت' if sc['status'] == 'READY' else '⏳ ' + (sc['waiting'][0] if sc['waiting'] else sc['setup_ar'])}"
            L.append(line)
        L.append("")
    L.append(f"### 🟡 تحت المراقبة — أقرب الأسهم لاكتمال الشروط ({len(result['watch'])})")
    for s in result["watch"][:C.MAX_WATCH_IN_REPORT]:
        L.append(watch_line(s))
    if not result["watch"]:
        L.append("لا يوجد.")
    L.append("")
    if C.ENABLE_MOMENTUM and result["session"] in ("pre", "regular", "post"):
        ttl = {"pre": "رادار البري ماركت", "regular": "رادار الزخم", "post": "رادار الأفتر ماركت"}[result["session"]]
        L.append(f"### ⚡ {ttl} ({len(result['momentum'])})")
        for s in result["momentum"][:C.MAX_MOMENTUM_IN_REPORT]:
            L.append(mom_line(s))
        if not result["momentum"]:
            L.append("لا يوجد سهم زخم يطابق الفلتر الآن.")
        L.append("")
    if result.get("errors"):
        L.append("ملاحظات تقنية: " + " | ".join(result["errors"][:3]))
    L.append("_تعليمي وليس توصية. ادخل بأوامر Limit داخل منطقة الطلب وقسّمها، وحدد الوقف قبل الدخول، "
             "وخذ ربحاً جزئياً عند أول مقاومة. إذا بطل السبب اخرج بدون عاطفة._")
    return "\n".join(L)
