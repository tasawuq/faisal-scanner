"""
تنبيهات تيليجرام فورية.
السر المطلوب في GitHub: TELEGRAM_BOT_TOKEN (من @BotFather).
TELEGRAM_CHAT_ID اختياري: إذا ما انضاف، يكتشفه الماسح تلقائياً من آخر شخص راسل البوت ويحفظه.
بدون تيليجرام: كل تنبيه يصير Issue في المستودع مع منشن لصاحبه → إشعار GitHub (التطبيق/الإيميل).
"""
import json
import logging
import os

import requests

log = logging.getLogger("faisal")
CHAT_FILE = os.path.join(".cache", "telegram_chat.json")  # خارج results (المستودع عام)


def _token():
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def _chat_id(token):
    cid = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if cid:
        return cid
    try:
        with open(CHAT_FILE, encoding="utf-8") as f:
            cid = str(json.load(f).get("chat_id", ""))
            if cid:
                return cid
    except Exception:  # noqa
        pass
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20).json()
        chats = [u.get("message", {}).get("chat", {}) for u in r.get("result", []) if u.get("message")]
        chats = [c for c in chats if c.get("type") == "private"]
        if chats:
            cid = str(chats[-1]["id"])
            os.makedirs(os.path.dirname(CHAT_FILE), exist_ok=True)
            with open(CHAT_FILE, "w", encoding="utf-8") as f:
                json.dump({"chat_id": cid}, f)
            return cid
    except Exception as e:  # noqa
        log.warning("telegram getUpdates failed: %s", e)
    return ""


def _gh():
    tok, repo = os.getenv("GITHUB_TOKEN", "").strip(), os.getenv("GITHUB_REPOSITORY", "").strip()
    return (tok, repo) if tok and repo and os.getenv("ALERT_ISSUES", "1") == "1" else None


def enabled() -> bool:
    return bool(_token()) or bool(_gh())


def gh_issue(text: str, title: str = None) -> bool:
    """تنبيه احتياطي: Issue مع منشن لصاحب المستودع → يوصله إشعار GitHub (التطبيق/الإيميل).
    ما نقفل الـIssue لأن القفل يرسل إيميل ثاني يلخبط."""
    g = _gh()
    if not g:
        return False
    tok, repo = g
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    if not title:
        lines = [x for x in text.split("\n") if x.strip()]
        title = lines[0] if lines else "تنبيه"
    body = "\n".join(f"{ln}  " for ln in text.split("\n"))  # سطر لكل معلومة (Markdown)
    try:
        r = requests.post(f"https://api.github.com/repos/{repo}/issues", headers=h, timeout=20,
                          json={"title": title[:240], "body": body + f"\n\n— @{repo.split('/')[0]}"})
        if not r.ok:
            log.warning("github issue alert failed: %s %s", r.status_code, r.text[:200])
        return r.ok
    except Exception as e:  # noqa
        log.warning("github issue alert failed: %s", e)
        return False


def send(text: str, title: str = None) -> bool:
    token = _token()
    if not token:
        return gh_issue(text, title)
    cid = _chat_id(token)
    if not cid:
        log.warning("telegram: no chat id yet — open the bot and press Start")
        return False
    ok = True
    for i in range(0, len(text), 3900):
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": cid, "text": text[i:i + 3900], "disable_web_page_preview": "true"},
                              timeout=20)
            ok = ok and r.ok
        except Exception as e:  # noqa
            log.warning("telegram send failed: %s", e)
            ok = False
    return ok


def fmt_price(x):
    if x is None:
        return "—"
    return f"{x:.2f}" if x >= 10 else f"{x:.3f}" if x >= 1 else f"{x:.4f}"


def _g(x):
    return "—" if x is None else f"{x:g}"


def open_time_riyadh(now_et=None) -> str:
    """وقت افتتاح السوق الرسمي بتوقيت الرياض (4:30 م صيفي / 5:30 م شتوي)."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    n = now_et or datetime.now(timezone.utc).astimezone(et)
    o = n.replace(hour=9, minute=30, second=0, microsecond=0).astimezone(ZoneInfo("Asia/Riyadh"))
    return f"{o.hour - 12 if o.hour > 12 else o.hour}:{o.minute:02d} م"


SETUP_SHORT = {"momentum": "زخم", "pivot": "ارتكاز"}


def alert_title(s: dict) -> str:
    return f"🚨 {s['ticker']} جاهز للدخول — {SETUP_SHORT.get(s.get('kind'), 'ارتكاز')}"


def alert_text(s: dict, when: str) -> str:
    """تنبيه سهم جاهز (Signal.to_dict()) — سطر لكل معلومة + وش تسوي."""
    p = fmt_price
    m = s.get("meta", {})
    mom = s.get("kind") == "momentum"
    seg = {"pre": "بري ماركت", "post": "أفتر ماركت"}.get(m.get("segment"))
    lines = [f"🚨 {s['ticker']} جاهز للدخول ({SETUP_SHORT.get(s.get('kind'), 'ارتكاز')})",
             f"الوقت: {when}" + (f" · {seg}" if seg else ""),
             f"النوع: {s['setup_ar']}",
             f"السعر الحالي: {p(s['price'])}",
             f"منطقة الدخول: {p(s.get('entry_low'))} – {p(s.get('entry_high'))}"
             + (f" (الفواب {p(m.get('vwap'))})" if mom and m.get("vwap") else "")]
    if s.get("stop") is not None:
        rp = s.get("risk_pct")
        lines.append(f"الوقف: {p(s['stop'])}" + (f" (خسارة {rp * 100:.1f}%)" if rp else ""))
    tg = [p(s[k]) for k in ("t1", "t2") if s.get(k)]
    if tg:
        lines.append("الأهداف: " + " ثم ".join(tg))
    extra = []
    if mom and m.get("change") is not None:
        extra.append(f"صاعد {m['change'] * 100:+.0f}% اليوم")
    if m.get("float"):
        extra.append(f"الأسهم الحرة {m['float'] / 1e6:.1f} مليون")
    if extra:
        lines.append(" · ".join(extra))
    if s.get("flags"):
        lines.append("⚠️ انتبه: " + " · ".join(s["flags"][:3]))
    if mom:
        act = (f"ادخل فقط بين {p(s.get('entry_low'))} و {p(s.get('entry_high'))} والسعر فوق الفواب، "
               f"والوقف {p(s.get('stop'))}. إذا ابتعد السعر عن المنطقة لا تطارده.")
    else:
        act = (f"حط أمر Limit بين {p(s.get('entry_low'))} و {p(s.get('entry_high'))}، "
               f"والوقف {p(s.get('stop'))}. لا تشتري فوق المنطقة.")
    if seg:
        act += f" (هذا سعر {seg} — الحركة ممكن تتغير مع الافتتاح {open_time_riyadh()})"
    lines += ["", "وش تسوي: " + act, "", "تعليمي وليس توصية."]
    return "\n".join(lines)


WATCH_HEAD = {"ZONE_CONFIRMED": "ثبت في منطقة الدخول (شمعة 15 دقيقة خضراء)", "IN_ZONE": "داخل منطقة الدخول (بدون تأكيد)",
              "BELOW_STOP": "نزل تحت الوقف", "BREAKOUT_HOLD": "اخترق وثبت",
              "RETEST_HOLD": "إعادة اختبار ناجحة لمستوى الاختراق"}


def watch_alert(ev: dict, when: str, seg: str = None) -> tuple:
    """تنبيه قائمة المتابعة/مستويات فيصل → (العنوان، النص)."""
    t, L, st = ev["ticker"], ev.get("levels") or {}, ev.get("state")
    lo, hi, stop, brk = L.get("demand_low"), L.get("demand_high"), L.get("stop"), L.get("breakout")
    head = WATCH_HEAD.get(st, ev.get("state_ar", ""))
    src = "مستويات فيصل" if "فيصل" in (ev.get("label") or "") else "متابعة Claude"
    title = f"🚨 {t} {head}" + (f" ({seg})" if seg else "")
    lines = [f"🚨 {t} — {head}", f"الوقت: {when}" + (f" · {seg}" if seg else ""),
             f"السعر الحالي: {fmt_price(ev.get('price'))}"]
    if st in ("RETEST_HOLD", "BREAKOUT_HOLD") and brk:
        lines.append(f"مستوى الاختراق: {_g(brk)}")
        lines.append(f"الدخول: قرب {_g(brk)} – {_g(round(brk * 1.03, 2))} (بدون مطاردة)")
        lines.append(f"الوقف: تحت {_g(brk)} مباشرة")
    else:
        if lo and hi:
            lines.append(f"منطقة الدخول: {_g(lo)} – {_g(hi)}")
        if stop:
            lines.append(f"الوقف: {_g(stop)}")
        if brk:
            lines.append(f"الاختراق: فوق {_g(brk)}")
    if L.get("targets"):
        lines.append("الأهداف: " + " ثم ".join(_g(x) for x in L["targets"]))
    lines.append(f"المصدر: {src}")
    wait_open = f"انتظر الافتتاح ({open_time_riyadh()})" if seg == "بري ماركت" else "انتظر جلسة بكرة" if seg else ""
    if st == "ZONE_CONFIRMED":
        tg = L.get("targets") or []
        act = (f"التأكيد تم: دخول ممكن داخل المنطقة {_g(lo)}–{_g(hi)} بدون مطاردة، والوقف {_g(stop)}."
               + (f" خذ جزء ربح عند {_g(tg[0])}." if tg else ""))
    elif st == "IN_ZONE":
        act = (f"{wait_open}، وبعدها " if wait_open else "") + \
              f"ادخل فقط إذا ثبت داخل المنطقة وطلعت شمعة 15 دقيقة خضراء. الوقف {_g(stop)}."
        if not wait_open:
            act = "لا تشتري على طول. " + act
    elif st == "BELOW_STOP":
        act = f"لا تدخل. وإذا عندك السهم: الوقف {_g(stop)} انكسر"
        act += f" في {seg} — إذا افتتح تحته اخرج." if seg else " — اخرج."
    elif st == "RETEST_HOLD":
        tg = [x for x in (L.get("targets") or []) if x > (ev.get("price") or 0)]
        act = (f"هذا الدخول اللي يفضله الملف بعد الاختراق («اشترِ الثبات أو إعادة الاختبار الناجحة»): "
               f"دخول قرب {_g(brk)}–{_g(round(brk * 1.03, 2))} بدون مطاردة، والوقف تحت {_g(brk)} مباشرة"
               f" (الوقف من تطبيق القاعدة، مو رقم فيصل)." + (f" الهدف {_g(tg[0])}." if tg else ""))
    elif st == "BREAKOUT_HOLD":
        act = f"دخول زخم قريب من {_g(brk)} بدون مطاردة، والوقف تحت {_g(brk)} مباشرة."
    else:
        act = ev.get("state_ar", "")
    if seg and st in ("ZONE_CONFIRMED", "BREAKOUT_HOLD"):
        act += (f" ⚠️ هذا في {seg}: السيولة أقل والسبريد أوسع، والحركة ممكن تنقلب "
                + (f"مع الافتتاح ({open_time_riyadh()})." if seg == "بري ماركت" else "في جلسة بكرة."))
    lines += ["", "وش تسوي: " + act, "", "تعليمي وليس توصية."]
    return title, "\n".join(lines)
