"""
فحص مباشر من البري ماركت (4:00 نيويورك) إلى نهاية الأفتر (20:00 نيويورك):
فحص الزخم + قائمة المتابعة كل دقيقة تقريباً داخل تشغيل واحد طويل (بدون الاعتماد على جدول GitHub).
التنبيهات الجديدة تُرسل فوراً (تيليجرام إذا موجود، وإلا إشعار GitHub).
    python -m faisal_scanner.live --minutes 345 --interval 60
"""
import argparse
import logging
import subprocess
import time as _time
from datetime import datetime, time, timezone

from .data_sources import ET, session_info
from .quick import run_quick

log = logging.getLogger("faisal")


def git_push(msg):
    subprocess.run(["git", "add", "results"], check=False)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        return
    subprocess.run(["git", "commit", "-q", "-m", msg], check=False)
    for _ in range(3):
        subprocess.run(["git", "pull", "-q", "--rebase", "-X", "theirs", "origin", "main"], check=False)
        if subprocess.run(["git", "push", "-q"], check=False).returncode == 0:
            return
        _time.sleep(3)


def ensure_running(workflow):
    """مراقبة متبادلة: كل حلقة تتأكد إن الحلقة الثانية شغالة وتشغّلها إذا وقفت (جدول GitHub والسلسلة يفشلون أحياناً)."""
    import os
    import requests
    tok, repo = os.getenv("GITHUB_TOKEN", ""), os.getenv("GITHUB_REPOSITORY", "")
    if not tok or not repo:
        return
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    base = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}"
    try:
        for st in ("in_progress", "queued"):
            r = requests.get(f"{base}/runs", headers=h, params={"status": st, "per_page": 1}, timeout=20)
            if r.ok and r.json().get("total_count", 0) > 0:
                return
        r = requests.post(f"{base}/dispatches", headers=h, json={"ref": "main"}, timeout=20)
        log.warning("watchdog: %s was not running → dispatched (%s)", workflow, r.status_code)
    except Exception as e:  # noqa
        log.warning("watchdog %s failed: %s", workflow, e)


def code_tree():
    r = subprocess.run(["git", "rev-parse", "HEAD:faisal_scanner"], capture_output=True, text=True, check=False)
    return r.stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=345)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--push-every", type=int, default=5, help="commit results every N iterations")
    ap.add_argument("--mode", choices=["quick", "full"], default="quick",
                    help="quick = زخم + مستويات فيصل كل دقيقة؛ full = فحص الارتكاز الكامل (كل الأسهم) بشكل متواصل")
    a = ap.parse_args(argv)
    if a.mode == "full":
        a.push_every = 1
    t_end = _time.time() + a.minutes * 60
    it, last_push = 0, 0
    tree0 = code_tree()  # إذا تحدث الكود في المستودع: نخرج وخطوة السلسلة تعيد التشغيل بالكود الجديد
    while _time.time() < t_end:
        now_et = datetime.now(timezone.utc).astimezone(ET)
        session, _ = session_info(now_et)
        if session == "closed":
            if now_et.weekday() >= 5 or now_et.time() >= time(20, 0):
                break
            _time.sleep(60)  # قبل 4:00 نيويورك: انتظار البري ماركت
            continue
        t0 = _time.time()
        new = []
        try:
            if a.mode == "full":
                from .main import run_once, write_outputs
                res = run_once() or {}
                if res:
                    write_outputs(res)
            else:
                res = run_quick() or {}
            new = res.get("new_alerts") or []
            log.info("live %s %s: movers=%s ready=%s new=%s (%.0fs)", session, f"{now_et:%H:%M}",
                     (res.get("debug") or {}).get("movers"), [x["ticker"] for x in res.get("ready", [])], new,
                     _time.time() - t0)
        except Exception as e:  # noqa
            log.warning("live iteration failed: %s", e)
        it += 1
        # كل ~10 دقائق: تأكد إن الحلقة الثانية شغالة
        if ((a.mode == "full") or it % 8 == 0) and now_et.time() < time(19, 45):
            ensure_running("faisal-live.yml" if a.mode == "full" else "faisal-pivot-live.yml")
        if new or it - last_push >= a.push_every:
            git_push(f"{'scan' if a.mode == 'full' else 'live'} {now_et:%H:%M} ET" + (f" NEW {','.join(new)}" if new else ""))
            last_push = it
            if tree0 and code_tree() != tree0:
                log.info("live: code updated in repo — restarting")
                break
        _time.sleep(max(5, a.interval - (_time.time() - t0)))
    git_push("live end")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
