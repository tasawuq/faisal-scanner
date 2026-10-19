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


def gh_issue(text: str) -> bool:
    """تنبيه احتياطي: Issue مغلق فوراً مع منشن لصاحب المستودع → يوصله إشعار GitHub."""
    g = _gh()
    if not g:
        return False
    tok, repo = g
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    lines = [x for x in text.split("\n") if x.strip()]
    title = " — ".join(lines[:2])[:240]
    try:
        r = requests.post(f"https://api.github.com/repos/{repo}/issues", headers=h, timeout=20,
                          json={"title": title, "body": text + f"\n\n@{repo.split('/')[0]}"})
        if r.ok:
            requests.patch(r.json()["url"], headers=h, json={"state": "closed"}, timeout=20)
        else:
            log.warning("github issue alert failed: %s %s", r.status_code, r.text[:200])
        return r.ok
    except Exception as e:  # noqa
        log.warning("github issue alert failed: %s", e)
        return False


def send(text: str) -> bool:
    token = _token()
    if not token:
        return gh_issue(text)
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


def alert_text(s: dict, when: str) -> str:
    """Arabic urgent alert for one ready signal (dict from Signal.to_dict())."""
    p = fmt_price
    m = s.get("meta", {})
    lines = [f"🚨 تنبيه عاجل — منهج فيصل ({when})", f"{s['ticker']} — {s['setup_ar']}", f"السعر: {p(s['price'])}"]
    if s.get("kind") == "momentum":
        lines.append(f"الدخول قرب: {p(s.get('entry_low'))}–{p(s.get('entry_high'))} | الفواب {p(m.get('vwap'))}")
    else:
        lines.append(f"الدخول (Limit): {p(s.get('entry_low'))}–{p(s.get('entry_high'))}")
    if s.get("stop") is not None:
        rp = s.get("risk_pct")
        lines.append(f"الوقف: {p(s['stop'])}" + (f" (−{rp * 100:.1f}%)" if rp else ""))
    tg = []
    if s.get("t1"):
        tg.append(f"{p(s['t1'])} ({s.get('t1_label', '')})")
    if s.get("t2"):
        tg.append(f"{p(s['t2'])}")
    if tg:
        lines.append("الأهداف: " + " ← ".join(tg))
    extra = []
    if m.get("float"):
        extra.append(f"فلووت {m['float'] / 1e6:.1f}M")
    if s.get("kind") == "momentum" and m.get("change") is not None:
        extra.append(f"{m['change'] * 100:+.0f}% اليوم")
    if extra:
        lines.append(" · ".join(extra))
    if s.get("flags"):
        lines.append("⚠️ " + " · ".join(s["flags"][:3]))
    lines.append("لا تطارد إذا ابتعد السعر عن منطقة الدخول. الوقف قبل الدخول. تعليمي وليس توصية.")
    return "\n".join(lines)
