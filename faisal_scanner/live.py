"""
فحص مباشر أثناء الجلسة: فحص الزخم + قائمة المتابعة كل دقيقة تقريباً داخل تشغيل واحد طويل
(بدون الاعتماد على جدول GitHub غير الموثوق). التنبيهات الجديدة تذهب لتيليجرام فوراً.
    python -m faisal_scanner.live --minutes 345 --interval 60
"""
import argparse
import logging
import subprocess
import time as _time
from datetime import datetime, time, timezone

from .data_sources import ET
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=345)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--push-every", type=int, default=5, help="commit results every N iterations")
    a = ap.parse_args(argv)
    t_end = _time.time() + a.minutes * 60
    it, last_push = 0, 0
    while _time.time() < t_end:
        now_et = datetime.now(timezone.utc).astimezone(ET)
        if now_et.time() >= time(16, 0):
            break
        t0 = _time.time()
        new = []
        if now_et.time() >= time(9, 30):
            try:
                res = run_quick() or {}
                new = res.get("new_alerts") or []
            except Exception as e:  # noqa
                log.warning("live iteration failed: %s", e)
        it += 1
        if new or it - last_push >= a.push_every:
            git_push(f"live {now_et:%H:%M} ET" + (f" NEW {','.join(new)}" if new else ""))
            last_push = it
        _time.sleep(max(5, a.interval - (_time.time() - t0)))
    git_push("live end")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
