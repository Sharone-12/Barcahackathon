"""One pass over the raw ATD, reduced to a small per-frame feature table and cached.

The raw tracking is 187 MB of text and 28 M floats; the model needs about twenty numbers per
frame. Parsing is therefore done once, vectorised, and cached as an npz, so iterating on the
possession model costs a second instead of two minutes.

    python features.py              # build (or reuse) the cache, then print a summary
    python features.py --rebuild    # after changing anything below

The parse is vectorised rather than streamed because every line of this export has exactly the
same shape -- verified, all 199319 lines carry 2 colons, 45 semicolons and 94 commas -- so turning
every separator into a comma makes the whole file one flat array of 142-column rows:

    frame : 46 x (x, y, speed) : ball (x, y, speed)

Coordinates are normalised 0-1 and the pitch is not square, so everything metric here is scaled by
GAMES[game]["pitch"] before any distance is compared. Ball speed is NaN throughout this export, so
it is differenced from the positions rather than read.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames  # noqa: E402

N_SLOTS = 46
ROW_WIDTH = 1 + N_SLOTS * 3 + 3

HOME, AWAY = 0, 1

CACHE = os.path.join(HERE, "cache", "features_{}_tactical.npz")

# A ball gap shorter than this is bridged by linear interpolation. The ball does not teleport, and
# at 25 fps half a second of occlusion is a tracking failure rather than a change in the game.
# Longer gaps are left missing on purpose: inventing a trajectory across three seconds of a
# crowded box would invent possession evidence along with it.
MAX_BALL_GAP = 13

# Velocity is differenced over this many frames rather than one. A single-frame difference on a
# noisy detection is mostly noise; a fifth of a second still resolves a pass.
VELOCITY_SPAN = 5

# Search range and sample for measure_speed_lag below. CENTRED_SPAN is the window of the
# lag-free reference derivative the reported speed is compared against.
MAX_SPEED_LAG = 12
CENTRED_SPAN = 4
LAG_SLOTS = 6
MIN_LAG_SAMPLES = 5000


def team_of_each_slot(meta_path):
    """(46,) int8: 0 where the track slot belongs to the home team, 1 where away.

    The first Team element in the FIFA/EPTS metadata is the home side -- the convention kloppy
    follows, and for this match that is Barcelona, which DATA.md states independently. Read from
    the XML rather than hard-coded, because the 24/22 split is data and not a property of the task.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if tid == team_ids[0] else AWAY for tid in slots], dtype=np.int8)


def parse_atd(raw_path, n_frames):
    """(players, ball) as (n, 46, 3) and (n, 3) float32, indexed by video frame.

    Every separator becomes a comma so the whole file parses as one flat numeric array. That is
    only safe because the row shape is invariant, which is checked here rather than assumed, and
    because the frame ids are 1..n contiguous, which is checked too -- a silent off-by-one in this
    function would move every label in the submission by one frame.
    """
    with open(raw_path, "rb") as handle:
        blob = handle.read()
    flat = np.fromstring(blob.translate(bytes.maketrans(b";:\n", b",,,")).strip(b","),
                         dtype=np.float64, sep=",")

    if flat.size % ROW_WIDTH:
        raise ValueError("{}: {} values is not a whole number of {}-column rows".format(
            raw_path, flat.size, ROW_WIDTH))
    rows = flat.reshape(-1, ROW_WIDTH)

    frame_ids = rows[:, 0].astype(np.int64)
    if len(rows) != n_frames or not np.array_equal(frame_ids, np.arange(1, len(rows) + 1)):
        raise ValueError(
            "{}: expected frame ids 1..{} contiguous, got {}..{} over {} rows".format(
                raw_path, n_frames, frame_ids[0], frame_ids[-1], len(rows)))

    players = rows[:, 1:1 + N_SLOTS * 3].reshape(-1, N_SLOTS, 3).astype(np.float32)
    ball = rows[:, 1 + N_SLOTS * 3:].astype(np.float32)
    return players, ball


def interpolate_short_gaps(values, seen, max_gap):
    """(filled, filled_seen) with linear interpolation across gaps no longer than max_gap.

    Gaps at the head and tail are never filled: there is no value on one side to interpolate from,
    and extrapolating a ball position is guessing rather than reconstructing.
    """
    filled = values.copy()
    filled_seen = seen.copy()
    known = np.flatnonzero(seen)
    if len(known) < 2:
        return filled, filled_seen

    left, right = known[:-1], known[1:]
    gap = right - left
    bridgeable = (gap > 1) & (gap - 1 <= max_gap)
    for start, stop in zip(left[bridgeable], right[bridgeable]):
        n = stop - start
        step = (values[stop] - values[start]) / n
        filled[start + 1:stop] = values[start] + step * np.arange(1, n, dtype=np.float32)
        filled_seen[start + 1:stop] = True
    return filled, filled_seen


def measure_speed_lag(px, py, pspeed, max_lag=MAX_SPEED_LAG):
    """Frames by which the ATD's reported speed field trails its own positions.

    The export ships a per-player speed alongside each position, and the two are not synchronous:
    the speed looks like a trailing-window derivative, so the value at frame t describes motion
    that happened several frames earlier. That matters more than it sounds, because every
    dead-ball cue in the model rests on collective speed, and a lagged cue produces late
    transitions -- which the ranking metric charges at the tight IoU thresholds.

    Measured rather than assumed, and measured without any ground truth: the ATD is simply checked
    against itself. A centred finite difference of the positions is a speed with no lag by
    construction, so the offset that best correlates the reported speed with it is the reported
    speed's own lag. On the delivered file the correlation peaks at 7 frames (0.964, against 0.943
    unshifted) and the curve is smooth and single-peaked either side.

    Computed on the most continuously tracked slots, because a slot that keeps dropping out
    contributes little but noise to a correlation over lags.
    """
    usable = np.isfinite(px) & np.isfinite(py) & np.isfinite(pspeed)
    slots = np.argsort(-usable.sum(axis=0))[:LAG_SLOTS]

    half = CENTRED_SPAN // 2
    scores = np.zeros(2 * max_lag + 1)
    counts = np.zeros(2 * max_lag + 1)

    for slot in slots:
        x, y, v = px[:, slot], py[:, slot], pspeed[:, slot]
        good = usable[:, slot]
        index = np.arange(half, len(x) - half)
        pair = good[index - half] & good[index + half]

        reference = np.full(len(x), np.nan)
        reference[index[pair]] = np.sqrt(
            (x[index + half] - x[index - half])[pair] ** 2
            + (y[index + half] - y[index - half])[pair] ** 2) / (2.0 * half / 25.0)

        for position, lag in enumerate(range(-max_lag, max_lag + 1)):
            shifted = shift(v, -lag)
            both = np.isfinite(shifted) & np.isfinite(reference)
            if both.sum() < MIN_LAG_SAMPLES:
                continue
            scores[position] += np.corrcoef(shifted[both], reference[both])[0, 1]
            counts[position] += 1

    valid = counts > 0
    if not valid.any():
        return 0
    mean_score = np.where(valid, scores / np.maximum(counts, 1), -np.inf)
    offset = int(np.arange(-max_lag, max_lag + 1)[int(np.argmax(mean_score))])

    # Returned as how far the report *trails*, a positive number, so the caller corrects with
    # shift(speed, lag) and cannot get the direction wrong. Inside the loop above the comparison
    # reads speed[t - offset], so a best offset of -7 means speed[t + 7] describes frame t, which
    # is a report running 7 frames late.
    return -offset


def shift(values, lag):
    """values shifted so that out[t] == values[t + lag], NaN where that falls off either end.

    Padded with NaN rather than wrapped: np.roll would move the end of the second half onto the
    start of the first, which is a different part of the match.
    """
    if lag == 0:
        return values
    out = np.full_like(values, np.nan, dtype=np.float32)
    if lag > 0:
        out[:-lag] = values[lag:]
    else:
        out[-lag:] = values[:lag]
    return out


def centred_speed(x, y, seen, span, length, width):
    """Ball speed in m/s from a centred finite difference over `span` frames.

    NaN wherever either endpoint is missing, so a speed is never differenced across an occlusion --
    the one place it would read as a huge spurious acceleration.
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    half = span // 2
    index = np.arange(n)
    low = np.clip(index - half, 0, n - 1)
    high = np.clip(index + half, 0, n - 1)
    usable = seen[low] & seen[high] & (high > low)

    dt = (high - low) / float(GAMES[GAME]["fps"])
    dx = (x[high] - x[low]) * length
    dy = (y[high] - y[low]) * width
    with np.errstate(invalid="ignore"):
        speed = np.sqrt(dx * dx + dy * dy) / np.where(dt > 0, dt, 1.0)
    out[usable] = speed[usable]
    return out


def build(game=GAME, data_dir=None):
    """The per-frame feature table as a dict of equal-length arrays, one entry per video frame."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]

    team = team_of_each_slot(meta_path)
    players, ball = parse_atd(raw_path, n_frames)

    px = players[:, :, 0] * length
    py = players[:, :, 1] * width
    pspeed = players[:, :, 2]
    tracked = np.isfinite(px) & np.isfinite(py)

    seen_raw = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    ball_x, seen = interpolate_short_gaps(
        np.where(seen_raw, ball[:, 0], np.nan).astype(np.float32), seen_raw, MAX_BALL_GAP)
    ball_y, _ = interpolate_short_gaps(
        np.where(seen_raw, ball[:, 1], np.nan).astype(np.float32), seen_raw, MAX_BALL_GAP)

    table = {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_seen_raw": seen_raw,
        "ball_seen": seen,
        "ball_speed": centred_speed(ball_x, ball_y, seen, VELOCITY_SPAN, length, width),
        "n_tracked": tracked.sum(axis=1).astype(np.int16),
    }

    # The reported speed trails the positions it was computed from, so it is realigned before
    # anything reads it -- see measure_speed_lag. The lag is measured from the file rather than
    # assumed, and is recorded in the cache so the correction is visible and auditable rather than
    # buried in an aggregate.
    lag = measure_speed_lag(px, py, pspeed)
    table["speed_lag"] = np.int16(lag)
    pspeed = shift(pspeed, lag) if lag else pspeed

    # Collective player motion: a dead-ball detector that does not depend on the ball at all.
    # During a stoppage players walk or stand; during play the distribution sits far higher. Both
    # are taken over tracked slots only, so a frame with few tracks is not dragged toward zero.
    has_speed = np.isfinite(pspeed) & tracked
    n_speed = has_speed.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        table["speed_mean"] = np.where(
            n_speed > 0,
            np.where(has_speed, pspeed, 0.0).sum(axis=1) / np.maximum(n_speed, 1),
            np.nan).astype(np.float32)
    table["speed_max"] = np.where(
        has_speed.any(axis=1), np.nanmax(np.where(has_speed, pspeed, -np.inf), axis=1), np.nan
    ).astype(np.float32)
    table["n_speed"] = n_speed.astype(np.int16)

    # Distance from the ball to the closest tracked player on each side -- the geometric evidence
    # for who has it. NaN where the ball is unknown, which is a third of the match, so every
    # consumer has to handle its absence rather than assume a number.
    far = np.float32(1e6)
    dx = px - (ball_x * length)[:, None]
    dy = py - (ball_y * width)[:, None]
    with np.errstate(invalid="ignore"):
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    dist[~tracked] = far
    dist[~seen] = far

    for code, name in ((HOME, "home"), (AWAY, "away")):
        nearest = dist[:, team == code].min(axis=1)
        table["d1_" + name] = np.where(nearest < far, nearest, np.nan)

    return table


def cache_path(game=GAME):
    return CACHE.format(game)


def load(game=GAME, data_dir=None, rebuild=False):
    """The feature table, from cache when it exists and by building it when it does not."""
    path = cache_path(game)
    if not rebuild and os.path.isfile(path):
        with np.load(path) as handle:
            return {key: handle[key] for key in handle.files}

    table = build(game, data_dir)
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    np.savez_compressed(path, **table)
    return table


if __name__ == "__main__":
    table = load(rebuild="--rebuild" in sys.argv)
    print("{} frames cached -> {}".format(
        len(table["ball_x"]), os.path.relpath(cache_path(), HERE)))
    print("  ball seen, raw     {:>6}  ({:.1f}%)".format(
        int(table["ball_seen_raw"].sum()), 100.0 * table["ball_seen_raw"].mean()))
    print("  ball seen, filled  {:>6}  ({:.1f}%)   gaps <= {} frames bridged".format(
        int(table["ball_seen"].sum()), 100.0 * table["ball_seen"].mean(), MAX_BALL_GAP))
    print("  tracked objects    mean {:.1f}".format(table["n_tracked"].mean()))
    print("  speed lag          {} frames ({:.2f} s), measured against the positions and "
          "corrected".format(int(table["speed_lag"]), int(table["speed_lag"]) / 25.0))
    for key in ("speed_mean", "speed_max", "ball_speed", "d1_home", "d1_away"):
        values = table[key][np.isfinite(table[key])]
        print("  {:<18} median {:7.3f}   p05 {:7.3f}  p95 {:7.3f}".format(
            key, float(np.median(values)), float(np.percentile(values, 5)),
            float(np.percentile(values, 95))))
