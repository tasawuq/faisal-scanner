"""
مصادر البيانات:
  - قائمة كل الأسهم الأمريكية + القيمة السوقية: Nasdaq screener API (احتياطي: ملفات nasdaqtrader)
  - الأسعار: Alpaca (إن وُجدت المفاتيح) وإلا Yahoo Finance
  - الفلووت والشورت والتقسيم العكسي: Yahoo (مع ذاكرة يومية)
  - الأخبار: Alpaca News أو Yahoo
"""
import json
import logging
import os
import time as _time
from datetime import datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from . import config as C

log = logging.getLogger("faisal")
ET = ZoneInfo("America/New_York")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")


def chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ============================================================ session
def session_info(now_et: datetime):
    if now_et.weekday() >= 5:
        return "closed", 0.0
    t = now_et.time()
    if time(4, 0) <= t < time(9, 30):
        return "pre", 0.0
    if time(9, 30) <= t < time(16, 0):
        mins = (now_et.hour * 60 + now_et.minute) - (9 * 60 + 30)
        return "regular", max(0.0, min(1.0, mins / 390.0))
    if time(16, 0) <= t < time(20, 0):
        return "post", 1.0
    return "closed", 0.0


# ============================================================ universe
def _num(x):
    if x is None:
        return np.nan
    s = str(x).replace("$", "").replace(",", "").replace("%", "").strip()
    if s in ("", "NA", "N/A", "--"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


BAD_NAME = ("warrant", " unit", " units", " right", " rights", "preferred", "depositary shares representing",
            "notes due", "% notes", "subordinated", "debenture", "trust preferred")


def fetch_universe_nasdaq() -> pd.DataFrame:
    rows = []
    hdr = {"User-Agent": UA, "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
           "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
    for ex in C.EXCHANGES:
        r = requests.get("https://api.nasdaq.com/api/screener/stocks",
                         params={"tableonly": "true", "limit": "10000", "offset": "0", "exchange": ex,
                                 "download": "true"}, headers=hdr, timeout=45)
        r.raise_for_status()
        data = (r.json() or {}).get("data") or {}
        rs = data.get("rows") or (data.get("table") or {}).get("rows") or []
        for d in rs:
            rows.append({
                "symbol": str(d.get("symbol", "")).strip().upper(),
                "name": d.get("name", ""),
                "price": _num(d.get("lastsale")),
                "pct_change": _num(d.get("pctchange")) / 100.0,
                "volume": _num(d.get("volume")),
                "market_cap": _num(d.get("marketCap")),
                "country": d.get("country", "") or "",
                "sector": d.get("sector", "") or "",
                "industry": d.get("industry", "") or "",
                "exchange": ex.upper(),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Nasdaq screener returned no rows")
    return _clean_universe(df)


def fetch_universe_nasdaqtrader() -> pd.DataFrame:
    rows = []
    r = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()
    for line in r.text.splitlines()[1:]:
        p = line.split("|")
        if len(p) < 7 or p[0].startswith("File Creation"):
            continue
        if p[3] == "Y" or p[6] == "Y":
            continue
        rows.append({"symbol": p[0], "name": p[1], "exchange": "NASDAQ"})
    r = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()
    exmap = {"N": "NYSE", "A": "AMEX"}
    for line in r.text.splitlines()[1:]:
        p = line.split("|")
        if len(p) < 7 or p[0].startswith("File Creation"):
            continue
        if p[4] == "Y" or p[6] == "Y" or p[2] not in exmap:
            continue
        rows.append({"symbol": p[0], "name": p[1], "exchange": exmap[p[2]]})
    df = pd.DataFrame(rows)
    for col in ("price", "pct_change", "volume", "market_cap"):
        df[col] = np.nan
    for col in ("country", "sector", "industry"):
        df[col] = ""
    return _clean_universe(df)


def _clean_universe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["symbol"].str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?")]
    low = df["name"].fillna("").str.lower()
    mask = np.ones(len(df), bool)
    for w in BAD_NAME:
        mask &= ~low.str.contains(w, regex=False).values
    return df[mask].drop_duplicates("symbol").reset_index(drop=True)


def get_universe() -> tuple:
    try:
        return fetch_universe_nasdaq(), "nasdaq-api"
    except Exception as e:  # noqa
        log.warning("Nasdaq screener failed (%s) — falling back to nasdaqtrader symbol files", e)
        return fetch_universe_nasdaqtrader(), "nasdaqtrader"


# ============================================================ Yahoo
def _yf_symbol(s):
    return s.replace(".", "-")


def _split_download(data, sym_yf):
    if data is None or len(data) == 0:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        if sym_yf in data.columns.get_level_values(0):
            d = data[sym_yf]
        elif sym_yf in data.columns.get_level_values(1):
            d = data.xs(sym_yf, axis=1, level=1)
        else:
            return None
    else:
        d = data
    d = d.rename(columns=lambda c: str(c).lower())
    need = ["open", "high", "low", "close", "volume"]
    if not all(c in d.columns for c in need):
        return None
    d = d[need].dropna(subset=["close"])
    return d if len(d) else None


class YahooData:
    name = "yahoo"

    def daily(self, symbols, period="1y") -> dict:
        import yfinance as yf
        out = {}
        for ch in chunks(symbols, 150):
            ys = [_yf_symbol(s) for s in ch]
            try:
                data = yf.download(ys, period=period, interval="1d", group_by="ticker", auto_adjust=False,
                                   actions=False, threads=True, progress=False)
            except Exception as e:  # noqa
                log.warning("yahoo daily chunk failed: %s", e)
                continue
            for s, y in zip(ch, ys):
                d = _split_download(data, y)
                if d is not None:
                    idx = pd.to_datetime(d.index)
                    if idx.tz is None:
                        idx = idx.tz_localize(ET)
                    d.index = idx
                    out[s] = d
            _time.sleep(0.5)
        return out

    def intraday(self, symbols, interval="15m", prepost=False) -> dict:
        import yfinance as yf
        out = {}
        for ch in chunks(symbols, 100):
            ys = [_yf_symbol(s) for s in ch]
            try:
                data = yf.download(ys, period="2d", interval=interval, group_by="ticker", auto_adjust=False,
                                   prepost=prepost, threads=True, progress=False)
            except Exception as e:  # noqa
                log.warning("yahoo intraday chunk failed: %s", e)
                continue
            for s, y in zip(ch, ys):
                d = _split_download(data, y)
                if d is not None:
                    idx = pd.to_datetime(d.index)
                    idx = idx.tz_localize("UTC") if idx.tz is None else idx
                    d.index = idx.tz_convert(ET)
                    out[s] = d
        return out

    def news(self, symbols, days=3) -> dict:
        import yfinance as yf
        out = {}
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        for s in symbols:
            try:
                items = yf.Ticker(_yf_symbol(s)).news or []
            except Exception:  # noqa
                continue
            lst = []
            for it in items:
                c = it.get("content", it)
                title = c.get("title") or it.get("title") or ""
                ts = c.get("pubDate") or c.get("displayTime")
                if ts:
                    dt = pd.to_datetime(ts, utc=True)
                elif it.get("providerPublishTime"):
                    dt = pd.to_datetime(it["providerPublishTime"], unit="s", utc=True)
                else:
                    continue
                if dt >= cutoff and title:
                    lst.append({"headline": title, "time": dt.isoformat()})
            if lst:
                out[s] = lst
            _time.sleep(0.2)
        return out


# ============================================================ Alpaca
class AlpacaData:
    name = "alpaca"
    BASE = "https://data.alpaca.markets"

    def __init__(self, key, secret, feed="sip"):
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Accept": "application/json"})
        self.feed = feed

    def _get(self, path, params):
        for attempt in range(5):
            r = self.s.get(self.BASE + path, params=params, timeout=45)
            if r.status_code == 429:
                _time.sleep(2 + attempt * 2)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Alpaca {path} {r.status_code}: {r.text[:200]}")
            return r.json()
        raise RuntimeError("Alpaca rate limit")

    def _end(self):
        if self.feed == "sip":
            return datetime.now(timezone.utc) - timedelta(minutes=C.ALPACA_SIP_DELAY_MIN)
        return None

    def bars(self, symbols, timeframe, start, end=None) -> dict:
        raw = {}
        for ch in chunks([s.replace("-", ".") for s in symbols], 100):
            params = {"symbols": ",".join(ch), "timeframe": timeframe, "start": start.isoformat().replace("+00:00", "Z"),
                      "limit": 10000, "adjustment": "split", "feed": self.feed, "sort": "asc"}
            if end is not None:
                params["end"] = end.isoformat().replace("+00:00", "Z")
            token = None
            while True:
                if token:
                    params["page_token"] = token
                j = self._get("/v2/stocks/bars", params)
                for sym, arr in (j.get("bars") or {}).items():
                    raw.setdefault(sym, []).extend(arr)
                token = j.get("next_page_token")
                if not token:
                    break
        out = {}
        for sym, arr in raw.items():
            if not arr:
                continue
            d = pd.DataFrame(arr)
            d.index = pd.to_datetime(d["t"], utc=True).dt.tz_convert(ET)
            d = d.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            out[sym] = d[["open", "high", "low", "close", "volume"]].astype(float)
        return out

    def daily(self, symbols, period="1y") -> dict:
        start = datetime.now(timezone.utc) - timedelta(days=400)
        return self.bars(symbols, "1Day", start, self._end())

    def intraday(self, symbols, interval="15m", prepost=False) -> dict:
        tf = {"5m": "5Min", "15m": "15Min"}.get(interval, interval)
        now_et = datetime.now(ET)
        start_et = datetime.combine(now_et.date(), time(4, 0), tzinfo=ET)
        out = self.bars(symbols, tf, start_et.astimezone(timezone.utc), self._end())
        if not prepost:
            for k, d in out.items():
                out[k] = d[[time(9, 30) <= ts.time() < time(16, 0) for ts in d.index]]
        return out

    def news(self, symbols, days=3) -> dict:
        out = {}
        start = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat().replace("+00:00", "Z")
        for ch in chunks(symbols, 20):
            try:
                j = self._get("/v1beta1/news", {"symbols": ",".join(ch), "start": start, "limit": 50, "sort": "desc"})
            except Exception as e:  # noqa
                log.warning("alpaca news failed: %s", e)
                continue
            for it in j.get("news", []):
                for sym in it.get("symbols", []):
                    if sym in ch:
                        out.setdefault(sym, []).append({"headline": it.get("headline", ""), "time": it.get("created_at")})
        return out


def merge_today_bar(daily: pd.DataFrame, intraday: pd.DataFrame, today) -> tuple:
    """Rebuild today's regular-session daily bar from intraday bars. Returns (df, partial_flag)."""
    if intraday is None or intraday.empty:
        return daily, False
    t = intraday[[ts.date() == today and time(9, 30) <= ts.time() < time(16, 0) for ts in intraday.index]]
    if t.empty:
        return daily, False
    bar = pd.DataFrame({"open": [t["open"].iloc[0]], "high": [t["high"].max()], "low": [t["low"].min()],
                        "close": [t["close"].iloc[-1]], "volume": [t["volume"].sum()]},
                       index=[pd.Timestamp(datetime.combine(today, time(0, 0)), tz=ET)])
    base = daily[[ts.date() < today for ts in daily.index]]
    return pd.concat([base, bar]), True


def yahoo_quotes(symbols) -> dict:
    """Batch quotes incl. pre/post-market (Yahoo v7 quote, ~150 symbols per request)."""
    from yfinance.data import YfData
    yd = YfData()
    out = {}
    for ch in chunks(symbols, 150):
        try:
            j = yd.get_raw_json("https://query1.finance.yahoo.com/v7/finance/quote",
                                params={"symbols": ",".join(_yf_symbol(s) for s in ch), "formatted": "false",
                                        "lang": "en-US", "region": "US"})
        except Exception as e:  # noqa
            log.warning("yahoo quote chunk failed: %s", e)
            continue
        back = {_yf_symbol(s): s for s in ch}
        for q in (j.get("quoteResponse") or {}).get("result") or []:
            s = back.get(q.get("symbol"))
            if not s:
                continue
            out[s] = {
                "reg_price": q.get("regularMarketPrice"),
                "reg_chg": (q.get("regularMarketChangePercent") or 0) / 100.0,
                "reg_vol": q.get("regularMarketVolume"),
                "prev_close": q.get("regularMarketPreviousClose"),
                "pre_price": q.get("preMarketPrice"),
                "pre_chg": (q.get("preMarketChangePercent") or 0) / 100.0,
                "post_price": q.get("postMarketPrice"),
                "post_chg": (q.get("postMarketChangePercent") or 0) / 100.0,
                "avg_vol": q.get("averageDailyVolume3Month") or q.get("averageDailyVolume10Day"),
                "state": q.get("marketState"),
            }
        _time.sleep(0.3)
    return out


NASDAQ_HDR = {"User-Agent": UA, "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
              "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}


def _find_num(obj, keys):
    """Search nested JSON for the first value whose key contains one of `keys` (case-insensitive)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(x in str(k).lower() for x in keys) and not isinstance(v, (dict, list)):
                n = _num(v)
                if np.isfinite(n):
                    return n
        for v in obj.values():
            r = _find_num(v, keys)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_num(v, keys)
            if r is not None:
                return r
    return None


def nasdaq_extended(symbols, market: str = "pre", raw_dump: str = None) -> dict:
    """Pre/after-hours consolidated volume & last price from Nasdaq (keyless). market: pre|post."""
    out, raws = {}, {}
    for s in symbols:
        try:
            r = requests.get(f"https://api.nasdaq.com/api/quote/{s}/extended-trading",
                             params={"markettype": market, "assetclass": "stocks", "time": 0},
                             headers=NASDAQ_HDR, timeout=20)
            j = r.json()
        except Exception as e:  # noqa
            log.debug("nasdaq ext %s failed: %s", s, e)
            continue
        if raw_dump is not None and len(raws) < 3:
            raws[s] = j
        data = (j or {}).get("data") or {}
        vol = _find_num(data, ("volume",))
        px = _find_num(data, ("consolidated", "lastsale", "price"))
        if vol is not None or px is not None:
            out[s] = {"volume": vol, "price": px}
        _time.sleep(0.15)
    if raw_dump is not None:
        try:
            os.makedirs(os.path.dirname(raw_dump), exist_ok=True)
            with open(raw_dump, "w", encoding="utf-8") as f:
                json.dump(raws, f, ensure_ascii=False, indent=1)
        except Exception:  # noqa
            pass
    return out


def load_daily_cache(key: str):
    import gzip, pickle
    path = os.path.join(C.CACHE_DIR, f"daily_{key}.pkl.gz")
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa
        return None


def save_daily_cache(key: str, daily: dict):
    import gzip, glob, pickle
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(C.CACHE_DIR, "daily_*.pkl.gz")):
        try:
            os.remove(old)
        except OSError:
            pass
    with gzip.open(os.path.join(C.CACHE_DIR, f"daily_{key}.pkl.gz"), "wb") as f:
        pickle.dump(daily, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_price_provider():
    if C.ALPACA_KEY and C.ALPACA_SECRET:
        return AlpacaData(C.ALPACA_KEY, C.ALPACA_SECRET, C.ALPACA_FEED)
    return YahooData()


# ============================================================ fundamentals (float / short / split)
FUND_KEYS = ("floatShares", "sharesOutstanding", "shortPercentOfFloat", "marketCap", "country", "sector",
             "industry", "lastSplitFactor", "lastSplitDate", "longName")


def fundamentals(symbols, cache_dir=C.CACHE_DIR) -> dict:
    import yfinance as yf
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "fundamentals.json")
    try:
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:  # noqa
        cache = {}
    today = datetime.now(ET).date().isoformat()
    out = {}
    fails = 0
    for s in symbols:
        c = cache.get(s)
        if c and c.get("_date") == today:
            out[s] = c
            continue
        if fails >= 15:  # Yahoo is blocking — keep stale cache
            if c:
                out[s] = c
            continue
        try:
            info = yf.Ticker(_yf_symbol(s)).info or {}
            rec = {k: info.get(k) for k in FUND_KEYS}
            rec["_date"] = today
            cache[s] = out[s] = rec
            fails = 0
        except Exception as e:  # noqa
            fails += 1
            log.debug("info %s failed: %s", s, e)
            if c:
                out[s] = c
        _time.sleep(0.15)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    return out
