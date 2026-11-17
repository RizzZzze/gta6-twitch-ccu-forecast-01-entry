"""Quantile forecast for Twitch CCU on a game's UTC release day, hours 2..24.

Version 3: the forecast is hour 1's print times a release-day shape fitted at
runtime from the case's own comparables (pool-median log-ratio per hour, with
a per-hour spread from the pool's dispersion), split into a steep-rise pool
and a rolling pool by the target's hour-1 level. When the target's own notes
say its launch already happened before the checkpoint (early access, or a
global simultaneous unlock at or before it), the shape's rising part is
halved: no major unlock step remains ahead, so the day is post-peak. The
house persistence forecast remains as the fallback rung: any failure in the
model path, or a pool too small to fit, degrades to persistence, and any
failure in a file's processing degrades to a valid fallback output. The
contract never changes: read every *.in in the input directory, write one
<id>.out per case, never crash, never emit an invalid shape.

Run by the grader as `python3 -P -s /bot/main.py` with no network. Standard
library only. Deterministic: fixed iteration order, no randomness, no clock.
"""

import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at LEVELS, as fixed by the competition rules.
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Log-spread per square-root hour of the persistence baseline.
LOG_SIGMA = 0.15
MAX_VIEWERS = 1_000_000_000.0
FIRST_HOUR = 1
FORECAST_HOURS = list(range(2, 25))


def round_half_up_2dp(value):
    """Round to 2 decimals, halves away from zero on the scaled value.

    Matches the house baseline's byte output: a whole result is returned as
    an int so the JSON writer prints `1240`, not `1240.0`.
    """
    scaled = float(value) * 100.0
    floored = math.floor(scaled)
    if scaled - floored >= 0.5:
        floored += 1
    rounded = floored / 100.0
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _clean_quantiles(raw):
    """Clamp to (0, 1e9], sort non-decreasing, round. Never empty, never NaN."""
    cleaned = []
    for item in raw:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        if number <= 0:
            number = 1.0
        if number > MAX_VIEWERS:
            number = MAX_VIEWERS
        cleaned.append(number)
    if not cleaned:
        cleaned = [1.0]
    cleaned.sort()
    while len(cleaned) < len(LEVELS):
        cleaned.append(cleaned[-1])
    return [round_half_up_2dp(v) for v in cleaned[: len(LEVELS)]]


def persistence_row(p1, hour):
    """One hour of the house persistence forecast around hour 1's print."""
    width = LOG_SIGMA * math.sqrt(hour - FIRST_HOUR)
    raw = [p1 * math.exp(z * width) for z in Z_SCORES]
    return _clean_quantiles(raw)


def _numeric(row, key):
    """Best-effort finite float from a row dict, or None."""
    if not isinstance(row, dict):
        return None
    try:
        number = float(row.get(key))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def extract_hour1_print(case):
    """Hour 1's print, with fallbacks. Always returns a finite value >= 1.

    Order: `known` row for hour 1, then the last `history_hourly` row, then
    the last `history` point, then 1. A print below 1 counts as 1.
    """
    known = case.get("known") if isinstance(case, dict) else None
    if isinstance(known, list):
        for row in known:
            if isinstance(row, dict) and row.get("hour") == FIRST_HOUR:
                viewers = _numeric(row, "viewers")
                if viewers is not None:
                    return max(1.0, viewers)
    for key in ("history_hourly", "history"):
        series = case.get(key) if isinstance(case, dict) else None
        if isinstance(series, list) and series:
            viewers = _numeric(series[-1], "viewers")
            if viewers is not None:
                return max(1.0, viewers)
    return 1.0


def forecast_hours(case):
    """Hours to forecast: the case's list when usable, else 2..24."""
    rows = case.get("forecast") if isinstance(case, dict) else None
    hours = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("hour"), int):
                hours.append(row["hour"])
    hours = sorted(set(hours))
    if len(hours) != 23 or hours != FORECAST_HOURS:
        return list(FORECAST_HOURS)
    return hours


# --- Model v2: pool-median release-day shape, fitted at runtime. ---
# The program fits from the case's own comparables: each past launch gives a
# normalised shape, ln(V_h / V_1) for h = 2..24. The target's forecast is
# hour 1's print times the pool median shape, with a per-hour spread from the
# pool's dispersion. Two pools: launches whose day rose steeply from hour 1
# (single global unlock after a quiet first hour) and the rest (rolling /
# already-live launches). The target joins the steep pool only when its own
# hour 1 is quiet in absolute terms; a big hour 1 means the launch is already
# under way whatever the unlock text says. The target launch itself is never
# in its own case's comparables, so the runtime fit is leave-one-out by
# construction. Change-point / region-ladder calibration from minute history
# (design doc section 4) is a later revision; v2 uses the pool shape only.
P1_SINGLE_CUT = 6000.0
RISE_CUT = 2.5
SIGMA_MULT = 1.0
TAIL_INFLATE = 1.15
SHRINK = 1.0
SIGMA_FLOOR = 0.15
EXCLUDE_EA_POOL = False
EXCLUDE_WEAK_POOL = True
# v3: damp the pool shape's late rise when the target's own schedule says the
# launch already happened before the checkpoint (early access, or a global
# simultaneous unlock at or before it), so no big unlock step remains ahead.
# Adopted: LOO ablation showed gains exactly on the mechanism-predicted cases
# with zero rolling regressions versus v2.
USE_DECAY_DAMP = True
DECAY_DAMP = 0.5
# Single-branch mixture spread: implemented and ablation-tested, NOT adopted.
# It left the pinball loss unchanged (within noise) while cutting coverage on
# three steep-pool cases, so it stays off; the code remains so the ablation
# stays reproducible.
SINGLE_MIXTURE = False
MIXTURE_W = 0.5

ISO_DATETIME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?Z")


def _parse_iso(text):
    """Parse an ISO-8601 UTC timestamp, or return None."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _checkpoint_time(case):
    """The case's checkpoint, or day_start + 1h, or None."""
    stamp = _parse_iso(case.get("checkpoint")) if isinstance(case, dict) else None
    if stamp is not None:
        return stamp
    start = _parse_iso(case.get("day_start")) if isinstance(case, dict) else None
    if start is not None:
        return start + timedelta(hours=1)
    return None


def _decay_evidence(case):
    """True when the target's notes say its launch already happened.

    Two schedule signals, both read from the case's own notes text:
    early access (a dated early-access launch, not a negated mention and not
    campaign-only early access), or a single/simultaneous global unlock
    scheduled at or before the checkpoint. Either means no major unlock step
    remains after the checkpoint, so the release day is post-peak. Unknown
    or unparseable notes mean no evidence: the pool shape stands unchanged.

    The unlock-phrase match is deliberately narrow: bare substring matching
    misfires on prose such as "single-player" next to an old timestamp, which
    falsely damped a real practice case. The phrase must be an unlock phrase
    (single/simultaneous followed by global/unlock/time/midnight/across/at),
    and a global unlock counts as done only when its own timestamp, within
    80 characters after the phrase, is at or before the checkpoint.
    """
    notes = case.get("notes") if isinstance(case, dict) else None
    if not isinstance(notes, str) or not notes:
        return False
    text = notes.lower()
    early = ("early access" in text or "early-access" in text)
    negated = ("no early access" in text or "no early-access" in text
               or "without early access" in text)
    if early and not negated and "campaign" not in text:
        return True
    if "midnight gmt" in text and "simultaneous" in text:
        return True
    phrase = re.compile(
        r"\bsingle\s+(global|simultaneous|unlock|time)\b"
        r"|\bsimultaneous\s+(global|midnight|across|at|unlock)\b")
    checkpoint = _checkpoint_time(case)
    if checkpoint is None:
        return False
    for hit in phrase.finditer(text):
        window = notes[hit.start():hit.start() + 80]
        for match in ISO_DATETIME_RE.finditer(window):
            stamp = datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                int(match.group(4)), int(match.group(5)),
                tzinfo=checkpoint.tzinfo,
            )
            if stamp <= checkpoint:
                return True
    return False


def _pool_shapes(case):
    """Normalised release-day shapes of the case's launch comparables.

    Returns a list of (shape, rise, weak, early_access). Skips the target's
    own-category history (not a launch) and comparables without a complete
    release day. `shape` holds 23 log-ratios, hours 2..24.
    """
    comparables = case.get("comparables") if isinstance(case, dict) else None
    if not isinstance(comparables, list):
        return []
    out = []
    for comp in comparables:
        if not isinstance(comp, dict):
            continue
        unlock = comp.get("unlock")
        unlock_text = unlock.lower() if isinstance(unlock, str) else ""
        if "not a launch" in unlock_text or "target-history" in unlock_text:
            continue
        day_start = _parse_iso(comp.get("day_start"))
        hourly = comp.get("hourly")
        if day_start is None or not isinstance(hourly, list):
            continue
        by_time = {}
        for point in hourly:
            stamp = _parse_iso(point.get("from")) if isinstance(point, dict) else None
            viewers = _numeric(point, "viewers") if isinstance(point, dict) else None
            if stamp is not None and viewers is not None:
                by_time[stamp] = viewers
        first = by_time.get(day_start)
        if first is None:
            continue
        p1 = max(1.0, first)
        shape = []
        complete = True
        for hour in range(2, 25):
            value = by_time.get(day_start + timedelta(hours=hour - 1))
            if value is None:
                complete = False
                break
            shape.append(math.log(max(1.0, value) / p1))
        if not complete:
            continue
        weak = "tier: weak" in unlock_text
        early_access = "early-access-contaminated" in unlock_text
        out.append((shape, max(shape), weak, early_access))
    return out


def _pool_fit(shapes):
    """Per-hour median log-ratio and spread (IQR / 1.349) of a shape pool."""
    medians = []
    sigmas = []
    for index in range(23):
        values = sorted(shape[index] for shape in shapes)
        medians.append(statistics.median(values))
        first_quartile = values[len(values) // 4]
        third_quartile = values[(3 * len(values)) // 4]
        sigmas.append(max(SIGMA_FLOOR, (third_quartile - first_quartile) / 1.349))
    return medians, sigmas


def _model_quantiles(p1, hour_index, median_log, sigma):
    """Five quantiles for one hour from the fitted shape and spread."""
    factors = [TAIL_INFLATE, 1.0, 1.0, 1.0, TAIL_INFLATE]
    centre = SHRINK * median_log
    raw = [
        p1 * math.exp(centre + z * factor * sigma * SIGMA_MULT)
        for z, factor in zip(Z_SCORES, factors)
    ]
    return _clean_quantiles(raw)


def _empirical_quantile(ordered, tau):
    """Linear-interpolation quantile of a sorted list of log-ratios."""
    if len(ordered) == 1:
        return ordered[0]
    position = tau * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _mixture_quantiles(p1, median_log, sigma, column):
    """Blend the pool's cross-sectional quantiles with Gaussian quantiles.

    In the steep pool the members unlock at different hours, so one hour's
    cross-section holds both pre-unlock (flat) and post-unlock (steep)
    members: the empirical quantiles are the scenario mixture over unlock
    times. Blending with the Gaussian quantiles shrinks the small-sample
    extremes toward the fitted centre and spread.
    """
    factors = [TAIL_INFLATE, 1.0, 1.0, 1.0, TAIL_INFLATE]
    centre = SHRINK * median_log
    raw = []
    for tau, z, factor in zip(LEVELS, Z_SCORES, factors):
        gaussian = centre + z * factor * sigma * SIGMA_MULT
        empirical = _empirical_quantile(column, tau)
        blended = (1.0 - MIXTURE_W) * gaussian + MIXTURE_W * empirical
        raw.append(p1 * math.exp(blended))
    return _clean_quantiles(raw)


def model_forecast(case, p1):
    """Pool-shape forecast, or None when the pool cannot support one."""
    pool = _pool_shapes(case)
    if len(pool) < 3:
        return None
    steep = [item for item in pool if item[1] > RISE_CUT]
    flat = [item for item in pool if item[1] <= RISE_CUT]
    if EXCLUDE_WEAK_POOL:
        steep = [item for item in steep if not item[2]]
        flat = [item for item in flat if not item[2]]
    if EXCLUDE_EA_POOL:
        steep = [item for item in steep if not item[3]]
        flat = [item for item in flat if not item[3]]
    chosen = steep if p1 < P1_SINGLE_CUT else flat
    single_branch = p1 < P1_SINGLE_CUT
    if len(chosen) < 3:
        chosen = flat if single_branch else steep
        single_branch = not single_branch
    if len(chosen) < 3:
        return None
    medians, sigmas = _pool_fit([item[0] for item in chosen])
    if USE_DECAY_DAMP and not single_branch and _decay_evidence(case):
        # No unlock step remains ahead: halve the shape's positive (rising)
        # part, keep the overnight dip untouched.
        medians = [m * (1.0 - DECAY_DAMP) if m > 0 else m for m in medians]
    shapes = [item[0] for item in chosen]
    hours = forecast_hours(case)
    rows = []
    for hour in hours:
        index = hour - 2
        if SINGLE_MIXTURE and single_branch:
            column = sorted(shape[index] for shape in shapes)
            quantiles = _mixture_quantiles(p1, medians[index], sigmas[index], column)
        else:
            quantiles = _model_quantiles(p1, index, medians[index], sigmas[index])
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def forecast_case(case):
    """Forecast one case: pool-shape model, persistence on any failure."""
    p1 = extract_hour1_print(case)
    try:
        modelled = model_forecast(case, p1)
    except Exception:
        modelled = None
    if modelled is not None:
        return modelled
    hours = forecast_hours(case)
    return {"hours": [{"hour": h, "q": persistence_row(p1, h)} for h in hours]}


def fallback_output():
    """Last-resort valid output: persistence around a print of 1."""
    return {"hours": [{"hour": h, "q": persistence_row(1.0, h)} for h in FORECAST_HOURS]}


def _serialise(payload):
    return json.dumps(payload, separators=(",", ":")) + "\n"


def process_file(path, output_dir):
    """Read one case file, write its forecast. Any failure -> fallback file.

    The write itself is guarded too: one unwritable output must not kill the
    remaining cases in the same run.
    """
    target = output_dir / (path.stem + ".out")
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(case, dict):
            raise ValueError("case is not a JSON object")
        payload = forecast_case(case)
    except Exception:
        payload = fallback_output()
    try:
        target.write_text(_serialise(payload), encoding="utf-8")
    except Exception:
        pass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        process_file(path, OUTPUT_DIR)


if __name__ == "__main__":
    main()
