"""Offline tests: synthetic patterns from the guide + data-source parsing with mocked responses."""
import json
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from faisal_scanner import demo
from faisal_scanner.data_sources import ET, AlpacaData, _split_download, fetch_universe_nasdaq, session_info
from faisal_scanner.momentum import analyze_momentum
from faisal_scanner.pivot import READY, WATCH, analyze_pivot

END = pd.Timestamp("2026-10-15")


def test_pump_and_broken_support_not_ready():
    for fn in (demo.scenario_pump, demo.scenario_failed, demo.scenario_extended):
        s = analyze_pivot("X", fn(END))
        assert s is None or s.status != READY


def test_ready_rate_on_random_walks_is_selective():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end="2026-10-15", periods=250).tz_localize(ET)
    ready = 0
    n = 600
    for _ in range(n):
        r = rng.normal(rng.normal(-0.002, 0.002), rng.uniform(0.03, 0.08), 250)
        c = 2.0 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]]
        h = np.maximum(o, c) * 1.02
        lo = np.minimum(o, c) * 0.98
        df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 3e5}, index=idx)
        s = analyze_pivot("R", df)
        ready += int(s is not None and s.status == READY)
    assert ready / n < 0.02


def test_momentum_vwap_pullback_ready():
    today = datetime(2026, 10, 15, 11, 0, tzinfo=ET)
    intr = demo.momentum_intraday(today.date())
    daily = demo.make_daily([1.6] + demo.seg(1.6, 2.0, 79), pd.Timestamp("2026-10-14"), seed=7, vol_base=300_000)
    m = analyze_momentum("M", intr, daily, today, "regular", 0.25)
    assert m is not None and m.status == READY and m.setup == "MOM_VWAP_PULLBACK"
    assert m.risk_pct <= 0.02


def test_session_info():
    assert session_info(datetime(2026, 10, 15, 10, 0, tzinfo=ET))[0] == "regular"
    assert session_info(datetime(2026, 10, 15, 7, 0, tzinfo=ET))[0] == "pre"
    assert session_info(datetime(2026, 10, 17, 10, 0, tzinfo=ET))[0] == "closed"


def test_yahoo_multiindex_split():
    idx = pd.date_range("2026-10-01", periods=3)
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    data = pd.DataFrame(np.ones((3, 12)), index=idx, columns=cols)
    d = _split_download(data, "BBB")
    assert list(d.columns) == ["open", "high", "low", "close", "volume"] and len(d) == 3
    cols2 = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["AAA"]])
    d2 = _split_download(pd.DataFrame(np.ones((3, 5)), index=idx, columns=cols2), "AAA")
    assert d2 is not None and len(d2) == 3


def test_alpaca_bars_pagination_and_parsing():
    pages = [
        {"bars": {"AAA": [{"t": "2026-10-13T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 1000}]},
         "next_page_token": "x"},
        {"bars": {"AAA": [{"t": "2026-10-14T04:00:00Z", "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 2000}],
                  "BBB": [{"t": "2026-10-14T04:00:00Z", "o": 3, "h": 3, "l": 3, "c": 3, "v": 10}]},
         "next_page_token": None},
    ]
    a = AlpacaData("k", "s", "sip")
    with mock.patch.object(a, "_get", side_effect=pages):
        out = a.bars(["AAA", "BBB"], "1Day", datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert len(out["AAA"]) == 2 and out["AAA"].index[-1].date().isoformat() == "2026-10-14"
    assert out["BBB"]["close"].iloc[0] == 3


def test_nasdaq_universe_parsing():
    payload = {"data": {"rows": [
        {"symbol": "ABCD", "name": "Abcd Inc. Common Stock", "lastsale": "$1.23", "pctchange": "5.5%",
         "volume": "1234567", "marketCap": "45000000.00", "country": "China", "sector": "Health Care",
         "industry": "Biotechnology"},
        {"symbol": "ABCDW", "name": "Abcd Inc. Warrant", "lastsale": "$0.05", "pctchange": "0%", "volume": "1",
         "marketCap": "", "country": "", "sector": "", "industry": ""},
        {"symbol": "XY^A", "name": "Pref", "lastsale": "$20", "pctchange": "0%", "volume": "1", "marketCap": "",
         "country": "", "sector": "", "industry": ""},
    ]}}
    resp = mock.Mock(status_code=200)
    resp.json.return_value = payload
    resp.raise_for_status = lambda: None
    with mock.patch("faisal_scanner.data_sources.requests.get", return_value=resp):
        u = fetch_universe_nasdaq()
    assert list(u["symbol"]) == ["ABCD"]
    row = u.iloc[0]
    assert row["price"] == pytest.approx(1.23) and row["pct_change"] == pytest.approx(0.055)
    assert row["market_cap"] == pytest.approx(45e6)


def _ext_bars(day, start, plan, vol):
    from datetime import timedelta
    t0 = datetime.combine(day, start, tzinfo=ET)
    rows, idx, px = [], [], plan[0]
    for i, c in enumerate(plan):
        rows.append([px, max(px, c) * 1.004, min(px, c) * 0.996, c, vol(i)])
        idx.append(pd.Timestamp(t0 + timedelta(minutes=5 * i)))
        px = c
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def test_premarket_momentum_pullback_flags_open_risk():
    from datetime import time as dtime
    now = datetime(2026, 10, 15, 7, 30, tzinfo=ET)
    plan = [2.0] + demo.seg(2.0, 2.60, 15) + demo.seg(2.60, 2.36, 10) + [2.37, 2.38, 2.385]
    intr = _ext_bars(now.date(), dtime(4, 0), plan, lambda i: 60_000 if i < 16 else 15_000)
    daily = demo.make_daily([1.6] + demo.seg(1.6, 2.0, 79), pd.Timestamp("2026-10-14"), seed=7, vol_base=300_000)
    m = analyze_momentum("P", intr, daily, now, "pre", 0.0, segment="pre")
    assert m is not None and m.meta["segment"] == "pre" and m.meta["change"] > 0.1
    assert any("بري ماركت" in f for f in m.flags)
    assert m.setup in ("MOM_VWAP_PULLBACK", "MOM_ABOVE_VWAP", "MOM_BREAKOUT_HOLD")


def test_afterhours_change_is_vs_regular_close():
    from datetime import time as dtime
    now = datetime(2026, 10, 15, 17, 0, tzinfo=ET)
    reg = _ext_bars(now.date(), dtime(9, 30), [2.0] * 78, lambda i: 20_000)
    post = _ext_bars(now.date(), dtime(16, 0), [2.0] + demo.seg(2.0, 2.5, 8), lambda i: 40_000)
    daily = demo.make_daily([1.6] + demo.seg(1.6, 2.0, 79), pd.Timestamp("2026-10-14"), seed=7, vol_base=300_000)
    m = analyze_momentum("Q", pd.concat([reg, post]), daily, now, "post", 1.0, segment="post")
    assert m is not None and abs(m.meta["change"] - 0.25) < 0.02
    assert any("أفتر ماركت" in f for f in m.flags)


def test_yahoo_quotes_parsing():
    from faisal_scanner import data_sources as ds
    payload = {"quoteResponse": {"result": [
        {"symbol": "ABCD", "regularMarketPrice": 1.0, "regularMarketChangePercent": 2.0, "preMarketPrice": 1.3,
         "preMarketChangePercent": 30.0, "averageDailyVolume3Month": 500000, "marketState": "PRE"},
        {"symbol": "BRK-B", "regularMarketPrice": 400.0}]}}
    with mock.patch("yfinance.data.YfData.get_raw_json", return_value=payload):
        q = ds.yahoo_quotes(["ABCD", "BRK.B"])
    assert q["ABCD"]["pre_chg"] == pytest.approx(0.30) and q["ABCD"]["pre_price"] == 1.3
    assert "BRK.B" in q


def test_premarket_without_volume_is_radar_only():
    from datetime import time as dtime
    now = datetime(2026, 10, 15, 7, 30, tzinfo=ET)
    plan = [2.0] + demo.seg(2.0, 2.60, 15) + demo.seg(2.60, 2.36, 10) + [2.37, 2.38, 2.385]
    intr = _ext_bars(now.date(), dtime(4, 0), plan, lambda i: 0)
    daily = demo.make_daily([1.6] + demo.seg(1.6, 2.0, 79), pd.Timestamp("2026-10-14"), seed=7, vol_base=300_000)
    m = analyze_momentum("Z", intr, daily, now, "pre", 0.0, segment="pre")
    assert m is not None and m.status == WATCH and m.meta["volume_known"] is False
    assert any("فوليوم" in w for w in m.waiting_for)


def test_faisal_watch_states():
    from faisal_scanner import faisal_watch as fw
    e = {"ticker": "GWAV", "demand_low": 2.70, "demand_high": 3.15, "stop": 2.55, "breakout": 3.70}
    assert fw.evaluate(e, 2.90)["state"] == "IN_ZONE"
    assert fw.evaluate(e, 2.40)["state"] == "BELOW_STOP"
    assert fw.evaluate(e, 3.40)["state"] == "WAITING"
    assert fw.evaluate(e, 3.80, last_close=3.60)["state"] == "BREAKOUT_PENDING"
    assert fw.evaluate(e, 3.80, hold_ok=True)["state"] == "BREAKOUT_HOLD"
    assert fw.evaluate(e, 2.90)["label"] == "متابعة"
    assert fw.evaluate({**e, "source": "تغريدة فيصل"}, 2.90)["label"] == "مستوى فيصل"


def test_faisal_watch_in_report(tmp_path, monkeypatch):
    import faisal_scanner.config as C
    from faisal_scanner import main, faisal_watch
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps({"stocks": [{"ticker": "DEMOA", "demand_low": 1.20, "demand_high": 1.30, "stop": 1.10,
                                           "note": "منطقة طلب", "source": "تغريدة فيصل"}]}), encoding="utf-8")
    monkeypatch.setattr(faisal_watch, "FILE", str(wl))
    monkeypatch.setattr(faisal_watch.load, "__defaults__", (str(wl),))
    monkeypatch.setattr(C, "RESULTS_DIR", str(tmp_path))
    res = main.run_once(demo=True)
    assert res["faisal"] and res["faisal"][0]["state"] == "IN_ZONE"
    # داخل المنطقة بدون شمعة تأكيد → لا تنبيه
    assert not any("مستوى فيصل" in a for a in res["new_alerts"])
    assert "أسهم فيصل" in main.write_outputs(res)


def test_premarket_volume_from_nasdaq_makes_volume_known():
    from datetime import time as dtime
    now = datetime(2026, 10, 15, 7, 30, tzinfo=ET)
    plan = [2.0] + demo.seg(2.0, 2.60, 15) + demo.seg(2.60, 2.36, 10) + [2.37, 2.38, 2.385]
    intr = _ext_bars(now.date(), dtime(4, 0), plan, lambda i: 0)
    daily = demo.make_daily([1.6] + demo.seg(1.6, 2.0, 79), pd.Timestamp("2026-10-14"), seed=7, vol_base=300_000)
    m = analyze_momentum("Z", intr, daily, now, "pre", 0.0, segment="pre", ext_volume=450_000)
    assert m is not None and m.meta["volume_known"] is True and m.meta["segment_volume"] == 450_000


def test_momentum_blocked_directly_under_daily_resistance():
    from datetime import time as dtime
    now = datetime(2026, 10, 15, 11, 0, tzinfo=ET)
    # يوم سابق قمته 2.03 — والسهم اليوم يخترق قمة الصباح عند ~2.00 تحت تلك المقاومة مباشرة
    daily = demo.make_daily([1.6] + demo.seg(1.6, 1.9, 79), pd.Timestamp("2026-10-14"), seed=7, vol_base=300_000)
    daily.iloc[-5] = [1.95, 2.03, 1.94, 1.96, 300_000]
    plan = [1.90] + demo.seg(1.90, 1.99, 10) + [1.98, 1.97, 1.975] + [2.005, 2.01, 2.012]
    intr = _ext_bars(now.date(), dtime(9, 30), plan, lambda i: 200_000)
    m = analyze_momentum("R", intr, daily, now, "regular", 0.3, segment="regular")
    if m is not None and m.setup == "MOM_BREAKOUT_HOLD":
        assert m.status == WATCH and any("مقاومة قريبة" in w for w in m.waiting_for)


def test_telegram_alert_text_and_disabled_without_token(monkeypatch):
    from faisal_scanner import notify
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert notify.enabled() is False and notify.send("x") is False
    s = {"ticker": "NCPL", "kind": "momentum", "setup_ar": "زخم — أول Pullback على الفواب", "price": 2.115,
         "entry_low": 2.09, "entry_high": 2.13, "stop": 2.078, "risk_pct": 0.017, "t1": 2.39, "t1_label": "قمة الجلسة",
         "t2": 2.66, "meta": {"vwap": 2.089, "float": 4.4e6, "change": 0.33}, "flags": []}
    txt = notify.alert_text(s, "19:00 الرياض")
    assert "NCPL" in txt and "2.078" in txt and "2.39" in txt


def test_watch_zone_needs_green_15m_confirmation():
    from datetime import time as dtime
    from faisal_scanner import faisal_watch as fw
    e = {"ticker": "W", "demand_low": 1.47, "demand_high": 1.50, "stop": 1.44, "breakout": 1.70, "source": "مستويات فيصل"}
    now = datetime(2026, 10, 19, 10, 2, tzinfo=ET)
    idx = [pd.Timestamp(datetime(2026, 10, 19, 9, 30, tzinfo=ET)), pd.Timestamp(datetime(2026, 10, 19, 9, 45, tzinfo=ET))]
    red = pd.DataFrame({"open": [1.55, 1.50], "high": [1.56, 1.51], "low": [1.49, 1.46], "close": [1.50, 1.47],
                        "volume": [1e5, 1e5]}, index=idx)
    green = red.assign(open=[1.55, 1.47], close=[1.50, 1.495])
    assert fw.zone_confirmed(red, 1.47, 1.50, now) is False
    assert fw.zone_confirmed(green, 1.47, 1.50, now) is True
    # the 10:00 bar is not complete at 10:02 → ignored
    assert fw.evaluate(e, 1.48, zone_ok=False)["alert"] is False
    ok = fw.evaluate(e, 1.48, zone_ok=True)
    assert ok["state"] == "ZONE_CONFIRMED" and ok["alert"] is True
    # outside the regular session: information only, no alert
    assert fw.evaluate(e, 1.40, alerts_on=False)["alert"] is False
    assert fw.evaluate(e, 1.48, zone_ok=True, alerts_on=False)["alert"] is False
    # 27 أكتوبر: مستويات المتابعة من عندي (مو فيصل) ما تعطي دخول بشمعة 15 دقيقة (ص90) — متابعة فقط
    mine = fw.evaluate({**e, "source": "متابعة Claude"}, 1.48, zone_ok=True)
    assert mine["alert"] is False and "ص86" in mine["state_ar"]
    assert fw.evaluate({**e, "source": "متابعة Claude"}, 1.40)["alert"] is True  # كسر الوقف معلومة لمن عنده السهم


def test_extended_session_confirmation_candle():
    from faisal_scanner.momentum import ext_confirmed
    from faisal_scanner import faisal_watch as fw
    now = datetime(2026, 10, 19, 8, 47, tzinfo=ET)
    t0 = datetime(2026, 10, 19, 8, 15, tzinfo=ET)
    idx = [pd.Timestamp(t0 + timedelta(minutes=5 * i)) for i in range(6)]  # 8:15..8:40 (8:45 bucket incomplete)
    up = pd.DataFrame({"open": [1.0, 1.01, 1.02, 1.00, 1.01, 1.02], "high": [1.02, 1.03, 1.04, 1.02, 1.03, 1.05],
                       "low": [0.99, 1.0, 1.01, 0.995, 1.0, 1.01], "close": [1.01, 1.02, 1.03, 1.01, 1.02, 1.04],
                       "volume": [1.0] * 6}, index=idx)
    assert ext_confirmed(up, now, v_now=1.0) is True              # 8:30 bucket: open 1.00 → close 1.04, above VWAP
    down = up.assign(close=[1.01, 1.02, 1.03, 1.0, 0.99, 0.98])
    assert ext_confirmed(down, now, v_now=1.0) is False           # red candle / below VWAP
    # watchlist in pre-market: stop needs a 15-min close below it
    e = {"ticker": "W", "demand_low": 1.47, "demand_high": 1.50, "stop": 1.44}
    b15 = pd.DataFrame({"open": [1.46], "high": [1.47], "low": [1.43], "close": [1.45], "volume": [0.0]},
                       index=[pd.Timestamp(datetime(2026, 10, 19, 8, 15, tzinfo=ET))])
    assert fw.stop_broken(b15, 1.44, now, "pre") is False
    ev = fw.evaluate(e, 1.43, stop_ok=fw.stop_broken(b15, 1.44, now, "pre"))
    assert ev["state"] == "BELOW_STOP" and ev["alert"] is False
    assert fw.evaluate(e, 1.43, stop_ok=True)["alert"] is True


def test_momentum_alert_rules(tmp_path, monkeypatch):
    """زخم صيني بدون تنبيه، وتنبيه زخم واحد لكل سهم بالجلسة (ACTU/CHOW 2026-10-19)."""
    import faisal_scanner.config as C
    from faisal_scanner.main import update_alert_state
    monkeypatch.setattr(C, "RESULTS_DIR", str(tmp_path))
    now = datetime(2026, 10, 19, 10, 30, tzinfo=ET)
    base = {"kind": "momentum", "price": 1.0, "stop": 0.98, "t1": 1.05, "setup_ar": "x", "meta": {"segment": "regular"}}
    chow = {**base, "ticker": "CHOW", "setup": "MOM_VWAP_PULLBACK", "flags": ["سهم صيني عالي التذبذب"]}
    a1 = {**base, "ticker": "ACTU", "setup": "MOM_BREAKOUT_HOLD", "flags": []}
    a2 = {**base, "ticker": "ACTU", "setup": "MOM_VWAP_PULLBACK", "flags": []}
    assert update_alert_state({"ready": [chow, a1], "faisal": []}, now) == ["ACTU"]
    assert update_alert_state({"ready": [a2], "faisal": []}, now) == []


def test_watch_breakout_retest_alert():
    """SVRE: اخترق 1.70 ووصل 1.96، ثم رجع لـ 1.72 وقفلت شمعة 15 دقيقة خضراء فوق 1.70 → تنبيه إعادة اختبار."""
    from faisal_scanner import faisal_watch as fw
    now = datetime(2026, 10, 19, 14, 32, tzinfo=ET)
    idx = [pd.Timestamp(datetime(2026, 10, 19, h, m, tzinfo=ET)) for h, m in ((10, 0), (13, 45), (14, 0), (14, 15))]
    b = pd.DataFrame({"open": [1.80, 1.80, 1.74, 1.71], "high": [1.96, 1.81, 1.75, 1.74],
                      "low": [1.79, 1.73, 1.70, 1.705], "close": [1.90, 1.74, 1.71, 1.73], "volume": [1e4] * 4}, index=idx)
    assert fw.retest_confirmed(b, 1.70, now) is True
    e = {"ticker": "SVRE", "demand_low": 1.50, "demand_high": 1.55, "stop": 1.45, "breakout": 1.70,
         "targets": [1.90, 2.06], "source": "مستويات فيصل"}
    ev = fw.evaluate(e, 1.73, retest_ok=True)
    assert ev["state"] == "RETEST_HOLD" and ev["alert"] is True
    red = b.assign(close=[1.90, 1.74, 1.71, 1.70], open=[1.80, 1.80, 1.74, 1.72])
    assert fw.retest_confirmed(red, 1.70, now) is False


def test_fill_missing_prices_fallback(monkeypatch):
    """Yahoo ما رجّع سعر البري → ناسداك ثم آخر شمعة 15 دقيقة حديثة؛ الشمعة القديمة ما تُستخدم."""
    import pandas as pd
    from datetime import datetime
    from faisal_scanner import data_sources as ds
    monkeypatch.setattr(ds, "nasdaq_extended", lambda syms, market="pre", raw_dump=None:
                        {"AAA": {"price": 1.23, "volume": 1000}} if "AAA" in syms else {})
    now = datetime(2026, 10, 20, 8, 0, tzinfo=ds.ET)
    fresh = pd.DataFrame({"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05], "volume": [10]},
                         index=[pd.Timestamp("2026-10-20 07:30", tz=ds.ET)])
    old = fresh.copy()
    old.index = [pd.Timestamp("2026-10-20 06:00", tz=ds.ET)]
    q, got = ds.fill_missing_prices({"CCC": {"pre_price": 2.0, "prev_close": 2.2}}, ["AAA", "BBB", "CCC", "DDD"],
                                    "pre", now, {"BBB": fresh, "DDD": old})
    assert q["AAA"]["pre_price"] == 1.23 and q["AAA"]["src"] == "nasdaq"
    assert q["BBB"]["pre_price"] == 1.05 and q["BBB"]["src"] == "bars"
    assert q["CCC"]["pre_price"] == 2.0 and "src" not in q["CCC"]
    assert "DDD" not in q
    assert got == {"nasdaq": 1, "bars": 1}


def test_base_breakout_without_retest_is_watch():
    """اختراق بدون إعادة اختبار = لا شراء للاختراق (ص20) → مراقبة."""
    df = demo.scenario_base_flag(END).iloc[:-2].copy()
    s = analyze_pivot("F", df)
    assert s is None or s.status != READY


def _bars15(day, rows):
    """rows: [(HH:MM, open, high, low, close, volume)] بتوقيت نيويورك."""
    idx = [datetime.combine(day, datetime.strptime(t, "%H:%M").time(), tzinfo=ET) for t, *_ in rows]
    return pd.DataFrame([r[1:] for r in rows], index=pd.DatetimeIndex(idx),
                        columns=["open", "high", "low", "close", "volume"])


def test_pivot_alert_regular_session_only(tmp_path, monkeypatch):
    """IPSC 22 أكتوبر: سهم ارتكاز جاهز نبه 04:02 في البري — التنبيه في الجلسة الرسمية فقط (ص10)."""
    import faisal_scanner.config as C
    from faisal_scanner import main
    monkeypatch.setattr(C, "RESULTS_DIR", str(tmp_path))
    now = datetime(2026, 10, 22, 4, 2, tzinfo=ET)
    s = {"ticker": "IPSC", "setup": "BASE_RETEST", "kind": "pivot", "price": 1.59, "stop": 1.473, "t1": 1.745,
         "entry_low": 1.55, "entry_high": 1.5965, "setup_ar": "علم", "meta": {}}
    assert main.update_alert_state({"ready": [s], "faisal": [], "session": "pre"}, now) == []
    assert main.update_alert_state({"ready": [s], "faisal": [], "session": "regular"},
                                   now.replace(hour=10)) == ["IPSC"]




# ============================================================ 27 أكتوبر: منهج فيصل حرفياً (method.py)
from faisal_scanner import method as M  # noqa: E402


def test_bounce_completes_p86_with_limit_ladder_and_stop_under_critical():
    """ص86: قاع ← إعادة اختبار وفشل الكسر (سحب سيولة) ← شمعة انعكاس بفوليوم = اكتمال الشروط السبعة؛
    الدخول 3 طلبات Limit داخل المنطقة (ص15، ص57)، والوقف تحت الدعم الحرج بتك (BJDX: كسر 1.67 خروج)."""
    s = analyze_pivot("A", demo.scenario_bounce(END))
    assert s is not None and s.status == READY and s.setup in ("SWEEP", "BOUNCE")
    ch = s.meta["p86"]["checks"]
    assert all(ch[k][0] for k in M.ORDER), ch
    lad = s.meta["ladder"]
    z_low, z_high = s.meta["zone"]
    assert len(lad["orders"]) == 3 and all(z_low - 0.01 <= o <= z_high + 1e-9 for o in lad["orders"])
    assert lad["stop"] < lad["critical"] <= z_low and abs(lad["critical"] - 1.17) < 0.01
    assert lad["risk"] <= M.MAX_LOSS and s.rr > 1.0
    assert not any("جاهز" in w for w in s.waiting_for)


def test_failed_support_lists_why_not_ready_with_pages():
    """ص17: كسر الدعم وبقي تحته = ضعف وليس سحب سيولة → متابعة، مع «لماذا غير جاهز» بصفحاتها و«متى يكون جاهزاً» (ص70)."""
    s = analyze_pivot("E", demo.scenario_failed(END))
    assert s is not None and s.status == WATCH
    assert any("ص17" in w and "وبقي تحته" in w for w in s.waiting_for)
    assert all("(ص" in w for w in s.waiting_for[:3])
    assert "ص86" in s.meta["when_ready"] and "ص54" in s.meta["when_ready"]


def test_sweep_then_reclaim_and_flag_retest_complete():
    """سحب سيولة ثم استرداد (ص17، ص86) و🚩 اختراق بعد قاعدة ثم إعادة اختبار ناجحة (ص79، ص68، ص86 دخول 3)."""
    s = analyze_pivot("B", demo.scenario_sweep(END))
    assert s is not None and s.setup == "SWEEP"
    # ص65: RSI الحالي 44 بعد الارتداد = «جزء من الارتداد حصل» → متابعة
    assert s.status == WATCH and any("RSI الحالي 44" in w for w in s.waiting_for)
    f = analyze_pivot("F", demo.scenario_base_flag(END))
    assert f is not None and f.setup == "BASE_RETEST" and f.status == READY
    assert f.stop < f.entry_low <= f.trigger_level <= f.entry_high and f.t1 > f.entry_high


def test_breakout_without_retest_never_signals():
    """ص20: «لا تشترِ الاختراق؛ اشترِ الثبات أو إعادة الاختبار الناجحة»."""
    df = demo.scenario_base_flag(END).iloc[:-2].copy()
    s = analyze_pivot("F", df)
    assert s is None or s.status != READY


def test_daily_analysis_uses_completed_bars_only():
    """ص54/ص90: التحليل على الشموع المكتملة؛ شمعة اليوم الجارية ما تكمل الشروط، والسعر الحالي يُستخدم للوقف فقط."""
    df = demo.scenario_bounce(END)
    closed = analyze_pivot("A", df)
    today = pd.DataFrame([[1.268, 1.27, 1.10, 1.12, 900_000.0]], columns=df.columns,
                         index=[df.index[-1] + pd.Timedelta(days=1)])
    s = analyze_pivot("A", pd.concat([df, today]), 0.5, True)
    assert s.meta["p86"]["bar_date"] == closed.meta["p86"]["bar_date"]
    assert s.price == pytest.approx(1.12) and s.status == WATCH
    assert any("تحت الدعم الحرج" in w for w in s.waiting_for)


def _gate_sig(t1):
    from faisal_scanner.pivot import Signal, apply_gate
    s = Signal(ticker="T", setup="BOUNCE", setup_ar="ارتداد", price=2.25, t1=t1)
    s.meta = {"zone": [2.24, 2.2624], "static_wait": [],
              "p86": {"critical": 2.18, "checks": {k: [True, "ok"] for k in M.ORDER}}}
    return apply_gate(s)


def test_direct_resistance_means_reward_not_above_risk():
    """ص82 «ارتداد أعلى من المخاطرة» + ص88 «الدخول تحت مقاومة مباشرة» (NERV 27 أكتوبر: الدخول 2.25 وقمة اليوم 2.335
    ووقف تحت 2.18): العائد من متوسط الطلبات لأول مقاومة لازم يتعدى الخسارة للوقف."""
    s = _gate_sig(2.30)
    assert s.status == WATCH and any("مقاومة مباشرة" in w for w in s.waiting_for)
    s = _gate_sig(2.40)
    assert s.status == READY and s.rr > 1


def test_four_hour_candle_completes_p86():
    """ص90 «قراءة الشموع تتم ع جميع الفريمات / اقربها 4 ساعات»: اليومي المكتمل ما فيه شمعة الانعكاس بعد،
    وشمعة 4 ساعات (9:30–13:30) مطرقة عند الدعم بفوليوم فوق المعتاد → تكتمل الشروط على فريم 4 ساعات."""
    from faisal_scanner.main import upgrade_4h
    df = demo.scenario_bounce(END).iloc[:-1].copy()
    df.index = df.index + pd.Timedelta(days=1)
    day = (df.index[-1] + pd.Timedelta(days=3)).date()
    s = analyze_pivot("A", df)
    assert s is not None and s.status == WATCH and not s.meta["p86"]["complete"]
    rows = [("09:30", 1.214, 1.216, 1.17, 1.18, 400_000)]
    rows += [(f"{h:02d}:{m:02d}", 1.18 + 0.006 * i, 1.19 + 0.006 * i, 1.175 + 0.006 * i, 1.186 + 0.006 * i, 60_000)
             for i, (h, m) in enumerate([(9, 45)] + [(h, m) for h in (10, 11, 12) for m in (0, 15, 30, 45)]
                                        + [(13, 0), (13, 15)])]
    bars = _bars15(day, rows)
    early = datetime.combine(day, datetime.strptime("13:20", "%H:%M").time(), tzinfo=ET)
    assert upgrade_4h(s, bars, early, 0) is False  # الشمعة ما اكتملت
    now = datetime.combine(day, datetime.strptime("13:45", "%H:%M").time(), tzinfo=ET)
    assert upgrade_4h(s, bars, now, 0) is True
    assert s.meta["p86"]["frame"] == "4 ساعات" and s.status == READY, s.waiting_for
    assert s.meta["ladder"]["critical"] <= 1.17 + 1e-9


def test_selection_follows_faisal_filter_and_template():
    """نفس مدى أسهم فيصل (≤8M، طلب المستخدم 27 أكتوبر)، ص6 Under 300M، والقالب (ص90–94): الجنسية، النشاط، التقسيم وافتتاحه وأعلاه، الشورت المتاح،
    الفلوت، عدد الأسهم، والسلوك السابق."""
    from faisal_scanner.main import enrich, data_line
    from faisal_scanner.pivot import Signal
    df = demo.scenario_bounce(END)
    split_ts = df.index[-40].timestamp()
    s = Signal(ticker="NERV", price=2.24)
    assert enrich(s, {"NERV": {"floatShares": 34.3e6, "sharesOutstanding": 47.7e6}}, {}) is False
    s = Signal(ticker="X", price=2.24)  # الفلوت مجهول وعدد الأسهم 47.7M → خارج مدى أسهم فيصل
    assert enrich(s, {"X": {"sharesOutstanding": 47.7e6}}, {}) is False
    s = Signal(ticker="B", price=1.6)  # BNKK (ص93): فلوت 7.84M داخل المدى، و9M خارجه
    assert enrich(s, {"B": {"floatShares": 7.84e6}}, {}) is True
    assert enrich(Signal(ticker="Z", price=1.6), {"Z": {"floatShares": 9e6}}, {}) is False
    assert enrich(Signal(ticker="L", price=2.2), {"L": {"sharesOutstanding": 45_878}}, {}) is False  # LGHL: أصغر من HTCR 541K
    s = Signal(ticker="U", price=1.6)
    assert enrich(s, {"U": {}}, {}) is True and any("غير معروفين" in fl for fl in s.flags)
    s = Signal(ticker="Y", price=1.5)
    f = {"Y": {"floatShares": 849e3, "sharesOutstanding": 1.2e6, "country": "Israel", "industry": "Software",
               "lastSplitFactor": "1:10", "lastSplitDate": split_ts, "marketCap": 3e6}}
    assert enrich(s, f, {}, df) is True
    assert any(t.startswith("تقسيم عكسي حديث 1:10") and "افتتح بعده" in t for t in s.tags)
    s.meta["runner"] = {"is_runner": True, "pct": 4.21, "from": 2.39, "from_date": "2025-10-15", "to": 12.46,
                        "to_date": "2025-10-21"}
    line = data_line(s.meta)
    for part in ("إسرائيلي", "Software", "قسم 1:10", "افتتاح", "اعلى", "الشورت المتاح: غير متوفر", "فلوت 0.85M",
                 "عدد الأسهم 1.20M", "سبق صعد من 2.39"):
        assert part in line, (part, line)
    c = Signal(ticker="C", price=1.5)
    enrich(c, {"C": {"floatShares": 2e6, "country": "China"}}, {})
    assert any("صيني" in fl for fl in c.flags)


def test_runner_is_100_percent_and_just_doubled_is_unsafe():
    """ص10: Former Runner = سبق انفجر 100%+ (+36% ما يكفي)؛ ص59: «توه مدبل… غير امن الان»."""
    idx = pd.bdate_range("2026-01-01", periods=60, tz=ET)
    base = np.full(60, 2.5)
    df = pd.DataFrame({"open": base, "high": base * 1.02, "low": base * 0.98, "close": base, "volume": 1e5}, index=idx)
    assert M.runner(df)["is_runner"] is False
    df.iloc[30, df.columns.get_loc("high")] = 2.5 * 1.36
    assert M.runner(df)["is_runner"] is False
    df.iloc[31, df.columns.get_loc("high")] = 12.46
    r = M.runner(df)
    assert r["is_runner"] and r["pct"] > 3
    assert M.just_doubled(df) is None
    df.iloc[-2, df.columns.get_loc("high")] = 5.2
    assert M.just_doubled(df) is not None


def test_pivot_alert_text_has_no_ready_word_and_explains():
    """نص الإشارة: بدون «جاهز»، فيه طلبات Limit 1/2/3، الخروج بكسر الدعم الحرج، وشروط ص86 كلها مع حالتها."""
    from faisal_scanner import notify
    s = analyze_pivot("A", demo.scenario_bounce(END))
    s.meta.update({"float": 2e6, "country_ar": "أمريكي"})
    d = s.to_dict()
    body = notify.alert_text(d, "17:00 الرياض")
    title = notify.alert_title(d)
    assert "جاهز" not in body and "جاهز" not in title
    assert "طلبات Limit" in body and "1)" in body and "3)" in body and "إغلاق شمعة يومية تحت الدعم الحرج" in body
    assert body.count("✅") == len(M.ORDER) and "ص86" in body and "الشورت المتاح" in body


def test_pivot_signal_once_per_completion_bar(tmp_path, monkeypatch):
    """الإشارة مرة وحدة لكل شمعة اكتمال (ما تتكرر بكرة على نفس الشمعة)، في الجلسة الرسمية أو بعد الإغلاق — لا بري (ص10)."""
    from faisal_scanner import main
    monkeypatch.setattr(main.C, "RESULTS_DIR", str(tmp_path))
    s = {"ticker": "VIP", "setup": "BOUNCE", "kind": "pivot", "price": 1.25, "stop": 1.19, "t1": 1.34,
         "entry_low": 1.22, "entry_high": 1.258, "setup_ar": "ارتداد", "meta": {"p86": {"bar_date": "2026-10-20"}}}
    d1 = datetime(2026, 10, 20, 16, 30, tzinfo=ET)
    assert main.update_alert_state({"ready": [s], "faisal": [], "session": "pre"}, d1.replace(hour=8)) == []
    assert main.update_alert_state({"ready": [s], "faisal": [], "session": "post"}, d1) == ["VIP"]
    assert main.update_alert_state({"ready": [s], "faisal": [], "session": "post"}, d1) == []
    d2 = datetime(2026, 10, 21, 10, 0, tzinfo=ET)
    assert main.update_alert_state({"ready": [s], "faisal": [], "session": "regular"}, d2) == []
    s2 = dict(s, meta={"p86": {"bar_date": "2026-10-21 09:30"}, "ladder": {"critical": 1.2}})  # نموذج جديد (دعم مختلف)
    assert main.update_alert_state({"ready": [s2], "faisal": [], "session": "regular"}, d2) == ["VIP"]


def test_weekly_list_is_watch_only(tmp_path, monkeypatch):
    """ص54 «اي سهم يتم طرحه ارتكاز غير جاهز للدخول، فقط متابعه»: القائمة بدون تنبيهات دخول، وفيها لماذا/متى والطلبات."""
    from faisal_scanner import weekly, faisal_watch
    monkeypatch.setattr(weekly, "FILE", str(tmp_path / "weekly.json"))
    monkeypatch.setattr(weekly.C, "RESULTS_DIR", str(tmp_path))
    daily = {"A": demo.scenario_bounce(END), "E": demo.scenario_failed(END), "B": demo.scenario_sweep(END)}
    pivots = [s for s in (analyze_pivot(t, d) for t, d in daily.items()) if s]
    w = weekly.build(pivots, daily, {}, datetime(2026, 10, 15, 18, 0, tzinfo=ET), news={})
    assert w["items"]
    for it in w["items"]:
        lv = it["levels"]
        assert len(lv["orders"]) == 3 and lv["stop"] < lv["critical"] <= lv["demand_low"]
        assert it["ready_when"] and it["p86"]["checks"]
    md = (tmp_path / "weekly.md").read_text(encoding="utf-8")
    assert "متابعة فقط" in md and "لماذا غير جاهز" in md or "متى يكون جاهزاً" in md
    ents = weekly.load_entries(str(tmp_path / "weekly.json"), max_age_days=10**6)
    for e in ents:
        for px in (e["demand_low"], e["demand_high"] * 1.3, (e["stop"] or e["demand_low"]) * 0.9):
            assert faisal_watch.evaluate(e, px, None, "", zone_ok=True, hold_ok=True, stop_ok=True)["alert"] is False


def test_demo_end_to_end(tmp_path, monkeypatch):
    import faisal_scanner.config as C
    from faisal_scanner import main
    monkeypatch.setattr(C, "RESULTS_DIR", str(tmp_path))
    res = main.run_once(demo=True)
    md = main.write_outputs(res)
    assert "DEMOA" in {s["ticker"] for s in res["ready"]}
    assert "اكتملت شروط ص86" in md and "جاهز الآن" not in md
    assert main.run_once(demo=True)["new_alerts"] == []


def test_signal_blocked_when_today_already_broke_the_stop():
    """LGHL 27 أكتوبر: شروط ص86 اكتملت على شمعة أمس، لكن اليوم نزل 2.00 تحت الوقف 2.24 ثم رجع 2.26 → الدعم انكسر،
    ما فيه إشارة (ص17 «كسر الدعم وبقي تحته = ضعف»، ص89 «كسر الدعم بعد الدخول = خروج»)."""
    df = demo.scenario_bounce(END)
    lad_stop = analyze_pivot("A", df).stop
    today = pd.DataFrame([[1.25, 1.27, lad_stop * 0.95, 1.20, 500_000.0]], columns=df.columns,
                         index=[df.index[-1] + pd.Timedelta(days=1)])
    s = analyze_pivot("A", pd.concat([df, today]), 0.8, True)
    assert s.status == WATCH and any("نزل اليوم" in w for w in s.waiting_for)


def test_same_setup_alerts_once_not_every_day(tmp_path, monkeypatch):
    """نفس النموذج (نفس الدعم الحرج) ما يتكرر تنبيهه كل يوم شمعة جديدة."""
    from faisal_scanner import main
    monkeypatch.setattr(main.C, "RESULTS_DIR", str(tmp_path))
    base = {"ticker": "VIP", "setup": "BOUNCE", "kind": "pivot", "price": 1.25, "stop": 1.19, "t1": 1.34,
            "entry_low": 1.22, "entry_high": 1.258, "setup_ar": "ارتداد"}
    s1 = dict(base, meta={"p86": {"bar_date": "2026-10-20"}, "ladder": {"critical": 1.2}})
    assert main.update_alert_state({"ready": [s1], "faisal": [], "session": "post"},
                                   datetime(2026, 10, 20, 16, 30, tzinfo=ET)) == ["VIP"]
    s2 = dict(base, meta={"p86": {"bar_date": "2026-10-21"}, "ladder": {"critical": 1.2}})
    assert main.update_alert_state({"ready": [s2], "faisal": [], "session": "post"},
                                   datetime(2026, 10, 21, 16, 30, tzinfo=ET)) == []
    s3 = dict(base, meta={"p86": {"bar_date": "2026-10-22"}, "ladder": {"critical": 1.1}})
    assert main.update_alert_state({"ready": [s3], "faisal": [], "session": "post"},
                                   datetime(2026, 10, 22, 16, 30, tzinfo=ET)) == ["VIP"]
