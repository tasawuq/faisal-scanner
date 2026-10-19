"""Offline tests: synthetic patterns from the guide + data-source parsing with mocked responses."""
import json
from datetime import datetime, timezone
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from faisal_scanner import demo
from faisal_scanner.data_sources import ET, AlpacaData, _split_download, fetch_universe_nasdaq, session_info
from faisal_scanner.momentum import analyze_momentum
from faisal_scanner.pivot import READY, WATCH, analyze_pivot

END = pd.Timestamp("2026-10-15")


def test_liquidity_sweep_bounce_is_ready():
    s = analyze_pivot("A", demo.scenario_bounce(END))
    assert s is not None and s.status == READY and s.setup in ("SWEEP", "BOUNCE")
    assert s.stop < s.support_low < s.price and s.t1 > s.price
    assert s.risk_pct <= 0.12


def test_w_neckline_retest_is_ready():
    s = analyze_pivot("C", demo.scenario_w(END))
    assert s is not None and s.status == READY and s.setup in ("W_RETEST", "W_BREAK")
    assert s.stop < s.trigger_level <= s.price


def test_w_forming_is_watch_with_neckline_condition():
    s = analyze_pivot("B", demo.scenario_sweep(END))
    assert s is not None and s.status == WATCH
    assert any("خط العنق" in w or "RSI" in w for w in s.waiting_for)


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


def test_demo_end_to_end(tmp_path, monkeypatch):
    import faisal_scanner.config as C
    from faisal_scanner import main
    monkeypatch.setattr(C, "RESULTS_DIR", str(tmp_path))
    res = main.run_once(demo=True)
    md = main.write_outputs(res)
    assert {"DEMOA", "DEMOC"} <= {s["ticker"] for s in res["ready"]}
    assert "جاهز الآن" in md
    data = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert data["new_alerts"]
    # second run the same day: no duplicate urgent alerts
    res2 = main.run_once(demo=True)
    assert res2["new_alerts"] == []


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


def test_near_daily_high_blocks_pivot_entry():
    """دخول ارتكاز تحت قمة أمس القريبة (أقرب من 3%) = تحت مقاومة مباشرة → مراقبة وليس جاهز."""
    import numpy as np
    import pandas as pd
    from faisal_scanner.pivot import _evaluate, Ctx, READY
    from faisal_scanner.levels import Zone
    n = 80
    idx = pd.date_range("2026-06-01", periods=n, freq="B", tz="America/New_York")
    close = np.linspace(3.0, 1.30, n)
    df = pd.DataFrame({"open": close * 1.01, "high": close * 1.02, "low": close * 0.99, "close": close,
                       "volume": np.full(n, 300_000.0)}, index=idx)
    df.iloc[-2, df.columns.get_loc("high")] = 1.34   # قمة أمس فوق السعر بـ 3%
    ctx = Ctx(df)
    z = Zone(1.28, 1.30, 3, n - 30, n - 5)
    cd = dict(code="SWEEP", zone=z, trigger=1.28, stop=1.225, entry=(1.28, 1.30), confirmed=True,
              rsi_lb=15, late=45, breakout=False)
    s = _evaluate("T", ctx, cd, {"MA20": None, "MA30": None, "MA50": None}, None)
    assert s is not None and abs(s.t1 - 1.34) < 1e-6
    assert any("مقاومة مباشرة" in w for w in s.waiting_for)


def test_watch_zone_needs_green_15m_confirmation():
    from datetime import time as dtime
    from faisal_scanner import faisal_watch as fw
    e = {"ticker": "W", "demand_low": 1.47, "demand_high": 1.50, "stop": 1.44, "breakout": 1.70}
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


def test_premarket_momentum_never_ready():
    from datetime import time as dtime
    now = datetime(2026, 10, 15, 7, 30, tzinfo=ET)
    plan = [2.0] + demo.seg(2.0, 2.60, 15) + demo.seg(2.60, 2.36, 10) + [2.37, 2.38, 2.385]
    intr = _ext_bars(now.date(), dtime(4, 0), plan, lambda i: 60_000 if i < 16 else 15_000)
    daily = demo.make_daily([1.6] + demo.seg(1.6, 2.0, 79), pd.Timestamp("2026-10-14"), seed=7, vol_base=300_000)
    m = analyze_momentum("P", intr, daily, now, "pre", 0.0, segment="pre", ext_volume=2_000_000)
    assert m is not None and m.status == WATCH
    assert any("الجلسة الرسمية" in w for w in m.waiting_for)
