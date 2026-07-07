"""
Analytics Engine — Week 5-6 Assignment 3


Takes raw GPS coordinate + timestamp + HR streams and computes:
  1. Cumulative distance (Haversine, point-to-point)
  2. Dynamic per-kilometer split times
  3. Heart Rate zone breakdown (Zone 1-5)
  4. Rolling averages for smoothing noisy GPS/HR data


"""

import json
import math
from datetime import datetime, timedelta



# 1. HAVERSINE DISTANCE


EARTH_RADIUS_M = 6371000  # meters


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two lat/lon points in meters.
    Standard Haversine formula.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (math.sin(d_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def compute_cumulative_distance(points):
    """
    points: list of dicts with keys 'lat', 'lon', 'timestamp' (ISO string or datetime)
    Returns: list of dicts, each point annotated with:
        - segment_distance_m: distance from previous point
        - cumulative_distance_m
        - cumulative_distance_km
    """
    annotated = []
    cumulative_m = 0.0

    for i, pt in enumerate(points):
        if i == 0:
            segment_m = 0.0
        else:
            prev = points[i - 1]
            segment_m = haversine_distance(prev['lat'], prev['lon'], pt['lat'], pt['lon'])

        cumulative_m += segment_m

        annotated.append({
            **pt,
            "segment_distance_m": round(segment_m, 2),
            "cumulative_distance_m": round(cumulative_m, 2),
            "cumulative_distance_km": round(cumulative_m / 1000, 4),
        })

    return annotated



# 2. PACE / SPLIT CALCULATIONS


def _parse_ts(ts):
    if isinstance(ts, datetime):
        return ts
    return datetime.fromisoformat(ts)


def format_pace(seconds_per_km):
    """Convert seconds/km into MM:SS string."""
    if seconds_per_km is None or math.isinf(seconds_per_km) or math.isnan(seconds_per_km):
        return "N/A"
    minutes = int(seconds_per_km // 60)
    secs = int(round(seconds_per_km % 60))
    if secs == 60:
        minutes += 1
        secs = 0
    return f"{minutes}:{secs:02d}"


def compute_splits(annotated_points):
    """
    Given points already annotated with cumulative_distance_km (from
    compute_cumulative_distance) and a timestamp field, compute dynamic
    per-kilometer splits.

    Returns a list of dicts:
        {km, split_time_seconds, split_pace (MM:SS/km), cumulative_time_seconds}
    Handles the final partial kilometer proportionally.
    """
    if not annotated_points:
        return []

    start_time = _parse_ts(annotated_points[0]['timestamp'])
    splits = []
    next_km_marker = 1
    last_km_time = start_time
    last_km_distance_m = 0.0

    total_distance_m = annotated_points[-1]['cumulative_distance_m']

    for i in range(1, len(annotated_points)):
        prev_pt = annotated_points[i - 1]
        curr_pt = annotated_points[i]

        prev_km = prev_pt['cumulative_distance_km']
        curr_km = curr_pt['cumulative_distance_km']

        # Crossed one or more km markers between prev and curr point
        while curr_km >= next_km_marker:
            t_prev = _parse_ts(prev_pt['timestamp'])
            t_curr = _parse_ts(curr_pt['timestamp'])
            d_prev = prev_pt['cumulative_distance_m']
            d_curr = curr_pt['cumulative_distance_m']

            marker_m = next_km_marker * 1000

            # Linear interpolation for exact time at marker_m
            if d_curr == d_prev:
                frac = 0.0
            else:
                frac = (marker_m - d_prev) / (d_curr - d_prev)
            marker_time = t_prev + (t_curr - t_prev) * frac

            split_seconds = (marker_time - last_km_time).total_seconds()
            splits.append({
                "km": next_km_marker,
                "split_time_seconds": round(split_seconds, 1),
                "split_pace": format_pace(split_seconds),  # exactly 1km per split
                "cumulative_time_seconds": round((marker_time - start_time).total_seconds(), 1),
            })

            last_km_time = marker_time
            last_km_distance_m = marker_m
            next_km_marker += 1

    # Final partial kilometer, if any distance remains
    remaining_m = total_distance_m - last_km_distance_m
    if remaining_m > 1.0:  # more than 1 meter left, worth reporting
        end_time = _parse_ts(annotated_points[-1]['timestamp'])
        split_seconds = (end_time - last_km_time).total_seconds()
        # normalize to a full-km-equivalent pace for comparison
        pace_per_km = split_seconds / (remaining_m / 1000) if remaining_m > 0 else None
        splits.append({
            "km": f"{next_km_marker} (partial, {round(remaining_m)}m)",
            "split_time_seconds": round(split_seconds, 1),
            "split_pace": format_pace(pace_per_km),
            "cumulative_time_seconds": round((end_time - start_time).total_seconds(), 1),
        })

    return splits



# 3. HEART RATE ZONES


# Standard % of Max HR zone thresholds
HR_ZONE_THRESHOLDS = {
    "Zone 1": (0.50, 0.60),   # Recovery
    "Zone 2": (0.60, 0.70),   # Aerobic base
    "Zone 3": (0.70, 0.80),   # Tempo / Aerobic threshold
    "Zone 4": (0.80, 0.90),   # Threshold / Anaerobic threshold
    "Zone 5": (0.90, 1.00),   # VO2 Max / Anaerobic
}


def classify_hr_zone(hr, max_hr):
    """Return the zone name for a single HR reading given max_hr."""
    pct = hr / max_hr
    for zone, (lo, hi) in HR_ZONE_THRESHOLDS.items():
        if lo <= pct < hi or (zone == "Zone 5" and pct >= hi):
            return zone
    return "Below Zone 1"  # very low HR, resting


def compute_hr_zone_breakdown(hr_points, max_hr, age=None):
    """
    hr_points: list of dicts with 'hr' and 'timestamp'
    max_hr: athlete's max heart rate (bpm). If None and age given, uses
            the 220-age estimate.
    Returns: dict with time (seconds) spent in each zone + percentages.
    """
    if max_hr is None:
        if age is None:
            raise ValueError("Provide either max_hr or age")
        max_hr = 220 - age

    zone_seconds = {z: 0.0 for z in HR_ZONE_THRESHOLDS}
    zone_seconds["Below Zone 1"] = 0.0

    for i in range(1, len(hr_points)):
        t_prev = _parse_ts(hr_points[i - 1]['timestamp'])
        t_curr = _parse_ts(hr_points[i]['timestamp'])
        dt = (t_curr - t_prev).total_seconds()

        # Attribute the interval to the zone of the earlier reading
        zone = classify_hr_zone(hr_points[i - 1]['hr'], max_hr)
        zone_seconds[zone] += dt

    total_seconds = sum(zone_seconds.values())

    breakdown = {}
    for zone, secs in zone_seconds.items():
        pct = (secs / total_seconds * 100) if total_seconds > 0 else 0.0
        breakdown[zone] = {
            "time_seconds": round(secs, 1),
            "time_mm_ss": format_pace(secs) if secs > 0 else "0:00",
            "percentage": round(pct, 1),
        }

    return {
        "max_hr_used": max_hr,
        "total_duration_seconds": round(total_seconds, 1),
        "zones": breakdown,
    }


# ---------------------------------------------------------------------------
# 4. ROLLING AVERAGES (no pandas dependency — pure python, but pandas-
#    equivalent logic is noted for reference)
# ---------------------------------------------------------------------------

def rolling_average(values, window):
    """
    Simple moving average over a list of numeric values.
    Equivalent to pandas: pd.Series(values).rolling(window, min_periods=1).mean()
    Uses min_periods=1 so output length == input length (no leading NaNs).
    """
    result = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        result.append(round(sum(chunk) / len(chunk), 3))
    return result


def smooth_hr_and_pace(annotated_points, hr_values, window=5):
    """
    Applies rolling average smoothing to HR values and instantaneous pace
    (derived from segment_distance_m / time delta) for cleaner charting.
    """
    smoothed_hr = rolling_average(hr_values, window)

    instantaneous_pace = []
    for i in range(len(annotated_points)):
        if i == 0:
            instantaneous_pace.append(None)
            continue
        t_prev = _parse_ts(annotated_points[i - 1]['timestamp'])
        t_curr = _parse_ts(annotated_points[i]['timestamp'])
        dt = (t_curr - t_prev).total_seconds()
        seg_m = annotated_points[i]['segment_distance_m']
        pace = (dt / (seg_m / 1000)) if seg_m > 0 else None
        instantaneous_pace.append(pace)

    valid_paces = [p for p in instantaneous_pace if p is not None]
    smoothed_pace_vals = rolling_average(valid_paces, window) if valid_paces else []

    # re-insert None placeholders where original was None
    smoothed_pace = []
    it = iter(smoothed_pace_vals)
    for p in instantaneous_pace:
        if p is None:
            smoothed_pace.append(None)
        else:
            smoothed_pace.append(next(it))

    return {
        "smoothed_hr": smoothed_hr,
        "smoothed_pace_seconds_per_km": [
            round(p, 1) if p is not None else None for p in smoothed_pace
        ],
    }


# ---------------------------------------------------------------------------
# ORCHESTRATION — Full pipeline, JSON-ready payload
# ---------------------------------------------------------------------------

def run_analytics_engine(raw_points, max_hr=None, age=None, rolling_window=5):
    """
    raw_points: list of dicts:
        {"lat": float, "lon": float, "timestamp": ISO str, "hr": int}

    Returns a single clean JSON-serializable dict with all computed metrics.
    """
    annotated = compute_cumulative_distance(raw_points)
    splits = compute_splits(annotated)
    hr_points = [{"hr": p["hr"], "timestamp": p["timestamp"]} for p in raw_points]
    hr_breakdown = compute_hr_zone_breakdown(hr_points, max_hr=max_hr, age=age)
    smoothing = smooth_hr_and_pace(annotated, [p["hr"] for p in raw_points], window=rolling_window)

    total_time_s = (_parse_ts(raw_points[-1]['timestamp']) - _parse_ts(raw_points[0]['timestamp'])).total_seconds()
    total_km = annotated[-1]['cumulative_distance_km']
    avg_pace = (total_time_s / total_km) if total_km > 0 else None

    payload = {
        "summary": {
            "total_distance_km": round(total_km, 3),
            "total_time_seconds": round(total_time_s, 1),
            "average_pace": format_pace(avg_pace),
        },
        "points": annotated,
        "splits": splits,
        "hr_zone_breakdown": hr_breakdown,
        "smoothing": smoothing,
    }
    return payload


if __name__ == "__main__":
    # ---- Demo / self-test with a small synthetic ~2.2km run ----
    base_time = datetime(2026, 7, 6, 6, 0, 0)
    demo_points = []
    lat, lon = 18.5204, 73.8567  # Pune
    hr = 130

    coord_deltas = [
        (0.0, 0.0), (0.0009, 0.0002), (0.0009, 0.0002), (0.0009, 0.0003),
        (0.0009, 0.0002), (0.0009, 0.0002), (0.0009, 0.0003), (0.0009, 0.0002),
        (0.0009, 0.0002), (0.0009, 0.0003), (0.0009, 0.0002), (0.0009, 0.0002),
        (0.0009, 0.0003), (0.0005, 0.0001),
    ]
    hr_walk = [128, 132, 138, 142, 145, 148, 150, 152, 149, 151, 154, 156, 153, 150]

    t = base_time
    for i, (dlat, dlon) in enumerate(coord_deltas):
        lat += dlat
        lon += dlon
        t = t + timedelta(seconds=30)
        demo_points.append({
            "lat": lat,
            "lon": lon,
            "timestamp": t.isoformat(),
            "hr": hr_walk[i],
        })

    result = run_analytics_engine(demo_points, age=21, rolling_window=3)
    print(json.dumps(result, indent=2))
