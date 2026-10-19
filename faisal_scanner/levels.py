"""Price structure: swing points, demand zones, resistances, candle patterns."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- swings
def _swings(values: np.ndarray, k: int, kind: str, edge: bool) -> list:
    n = len(values)
    out = []
    for i in range(1, n):
        right = n - 1 - i
        if right < (1 if edge else k):
            continue
        lo, hi = max(0, i - k), min(n, i + k + 1)
        win = values[lo:hi]
        v = values[i]
        is_ext = v <= win.min() + 1e-12 if kind == "low" else v >= win.max() - 1e-12
        if not is_ext:
            continue
        if out and i - out[-1] <= k:
            prev = out[-1]
            better = values[i] < values[prev] if kind == "low" else values[i] > values[prev]
            if better:
                out[-1] = i
            continue
        out.append(i)
    return out


def swing_lows(low: np.ndarray, k: int = 3, edge: bool = False) -> list:
    return _swings(np.asarray(low, float), k, "low", edge)


def swing_highs(high: np.ndarray, k: int = 3, edge: bool = False) -> list:
    return _swings(np.asarray(high, float), k, "high", edge)


# ---------------------------------------------------------------- zones
@dataclass
class Zone:
    low: float
    high: float
    touches: int
    first_idx: int
    last_idx: int
    is_bottom: bool = False  # أدنى قاع في الفترة (قاع تاريخي/سنوي)

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0


def build_zones(points, tol: float) -> list:
    """points: list of (idx, price). Cluster prices within `tol` into demand zones."""
    pts = sorted(points, key=lambda x: x[1])
    zones, cur = [], []

    def close_group(g):
        ps = [p for _, p in g]
        idxs = [i for i, _ in g]
        lo = min(ps)
        hi = max(max(ps), lo * 1.01)
        return Zone(lo, hi, len(g), min(idxs), max(idxs))

    for idx, p in pts:
        if cur and p > cur[0][1] * (1 + tol):
            zones.append(close_group(cur))
            cur = []
        cur.append((idx, p))
    if cur:
        zones.append(close_group(cur))
    return zones


def count_visits(low: np.ndarray, close: np.ndarray, z: Zone, start: int, end: int, tol: float) -> int:
    """Episodes where price dipped into the zone and did not close far below it."""
    episodes, inside_prev, gap = 0, False, 0
    for i in range(start, end):
        inside = low[i] <= z.high * (1 + tol / 2) and low[i] >= z.low * (1 - tol) and close[i] >= z.low * (1 - 0.01)
        if inside and not inside_prev and (gap >= 2 or episodes == 0):
            episodes += 1
        if inside:
            gap = 0
        else:
            gap += 1
        inside_prev = inside
    return episodes


# ---------------------------------------------------------------- resistances
def resistance_levels(df: pd.DataFrame, price: float, lookback: int, k: int, ma_values: dict,
                      floor_pct: float = 0.03) -> list:
    """Return sorted list of (level, label_ar) above price — targets / walls.
    floor_pct: ignore levels closer than this above price (0.03 for pivot targets; smaller for a full level map)."""
    d = df.iloc[-lookback:]
    o, h, l, c = (d[x].values for x in ("open", "high", "low", "close"))
    floor = price * (1 + floor_pct)
    cands = []
    for i in swing_highs(h, k=k, edge=True):
        if h[i] >= floor:
            cands.append((float(h[i]), "قمة سابقة"))
    # رأس الشمعة الهابطة الكبيرة (GDHG)
    for i in range(len(d)):
        if c[i] < o[i] and (o[i] - c[i]) / max(o[i], 1e-9) >= 0.08 and o[i] >= floor:
            cands.append((float(o[i]), "رأس شمعة هابطة"))
    # فجوات هابطة لم تُغطَّ (ADIL)
    for i in range(1, len(d)):
        if h[i] < l[i - 1] * 0.98:
            top = float(l[i - 1])
            if top >= floor and (i + 1 >= len(d) or h[i + 1:].max() < top):
                cands.append((top, "فجوة لم تُغطَّ"))
    for name, v in ma_values.items():
        if v is not None and np.isfinite(v) and v >= floor:
            cands.append((float(v), name))
    if not cands:
        return []
    cands.sort(key=lambda x: x[0])
    merged = []
    for lvl, lab in cands:
        if merged and lvl <= merged[-1][0] * 1.03:
            if lab not in merged[-1][1]:
                merged[-1] = (merged[-1][0], merged[-1][1] + " + " + lab)
            continue
        merged.append((lvl, lab))
    return merged


# ---------------------------------------------------------------- candles
def candle(o, h, l, c, po=None, pc=None) -> dict:
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    lower = min(o, c) - l
    upper = h - max(o, c)
    close_pos = (c - l) / rng
    out = {
        "hammer": lower >= 2 * max(body, rng * 0.05) and lower / rng >= 0.5 and upper / rng <= 0.3,
        "engulfing": po is not None and c > o and pc < po and c >= po and o <= pc,
        "lower_wick": lower / rng >= 0.45 and close_pos >= 0.55,
        "strong_green": c > o and close_pos >= 0.6,
        "upper_wick_ratio": upper / rng,
        "close_pos": close_pos,
        "green": c > o,
    }
    out["bullish_reversal"] = out["hammer"] or out["engulfing"] or out["lower_wick"]
    return out


def candle_name(info: dict) -> str:
    if info.get("engulfing"):
        return "ابتلاعية شرائية (Engulfing)"
    if info.get("hammer"):
        return "مطرقة (Hammer)"
    if info.get("lower_wick"):
        return "ذيل سفلي طويل (رفض كسر)"
    if info.get("strong_green"):
        return "شمعة خضراء بإغلاق قوي"
    return ""
