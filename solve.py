"""Frame-level ball possession as a three-state HMM decoded with Viterbi.

    python features.py       # once, builds the cached feature table
    python solve.py          # writes evaluation/results/hmm.csv

Why a state model rather than a smoothed per-frame classifier
-------------------------------------------------------------
The ranking metric scores whole possessions matched by temporal IoU, so what it rewards is a
signal whose segments line up with the truth's segments -- not one whose individual frames are
right. The reference baseline gets that structure from a centred majority vote, which is a
smoothing window standing in for a duration prior: too narrow and possessions fragment, too wide
and they merge, and the best width is not discoverable from the public five minutes.

An HMM puts the duration prior where it belongs, in the transition matrix. Staying in a state is
cheap and leaving it costs a fixed log-penalty, so a possession survives a few seconds of contrary
evidence while a sustained change still flips the state. The segments come out whole because the
model is built on the assumption that they are whole.

What the model reads
--------------------
Three states, H / A / D, and the emission at each frame splits into two independent questions:

  is the ball in play       collective player speed (the strongest cue present on every frame),
                            whether the tracker has the ball at all, whether the ball is off the
                            pitch, and whether Smart Tagging marks a restart nearby

  if it is, whose is it     which side has the closest tracked player to the ball, and which team
                            Smart Tagging's phase tags point at

Both halves are needed because neither is available everywhere: the ball is missing on 41% of
graded frames, the tagging covers 68% of them, and the two gaps do not coincide.

On the constants
----------------
The provided game may not be tuned against -- not on the published five minutes and not on
anything derived from the private truth -- so every number below comes from one of three places,
and which one is stated at its definition:

  football and physics      a possession lasts of the order of twelve seconds and a stoppage
                            twenty-five; ball-in-play is a little under 60% of the wall clock;
                            control needs the ball within a few metres; a ball off the pitch is
                            out of play by the Laws of the Game

  unsupervised, this match   the two collective-speed distributions. The dead one is seeded by the
                            Laws of the Game and the alive one recovered by deconvolution, with
                            no labels anywhere -- see fit_speed_model

  self-calibrating          one offset, pinned so that the decoded dead share matches the
                            externally known base rate -- see calibrate_offset

The published five minutes are used once, at the end, to check the pipeline is correct -- frames
aligned, file well formed, labels meaning what they are supposed to. No parameter here was chosen
by scoring against them, because 17 possessions cannot choose one: EVALUATION.md shows the
baseline's own smoothing window being picked wrongly that way.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES, load as load_tagging  # noqa: E402

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])
OUT = os.path.join(HERE, "evaluation", "results", "hmm.csv")

FPS = GAMES[GAME]["fps"]

# -- Smart Tagging semantics -------------------------------------------------------------------
# The phases come in attack/defend mirror pairs -- PROGRESSION Barcelona and DEFENDING IN MIDDLE
# THIRD Atletic Club appear 47 times each with the same median duration, and so on down the list.
# So an offensive tag and the defensive tag facing it are two views of one phase, and both point
# at the same team having the ball. Reading them as a vote rather than as a label is what makes
# the false positives survivable.
OFFENSIVE = frozenset((
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
    "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX"))
# For these the tagged team is the one *without* the ball, so the vote goes to its opponent.
DEFENSIVE = frozenset((
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION"))
# Neither of these is a possession cue. A set piece is a restart, so it implies a stoppage just
# before it; a goal is followed by a celebration and a kickoff, which is a long dead stretch.
# Both are read as evidence about the ball being out of play, never about who holds it.
STOPPAGE_CODES = frozenset(("SET PIECES", "GOALS"))
TEAM_CODE = {"Barcelona": H, "Atletic Club": A}

# -- parameters: football and physics ----------------------------------------------------------
# How sticky each state is, which is the whole duration prior. Derived from how elite football
# divides a match rather than from this one: ball-in-play time runs a little under 60% of the
# wall clock, a match contains of the order of 300 possessions, and of the order of 110 stoppages
# (throw-ins, free kicks, goal kicks, corners, offsides, goals, substitutions). Against the 106
# graded minutes given in DATA.md that is roughly 60 minutes alive over 300 possessions and 46
# minutes dead over 110 stretches. They are orders of magnitude, not measurements: the penalty
# enters as a log, so the decode is insensitive to a second or two either way.
POSSESSION_SECONDS = 12.0
DEAD_SECONDS = 25.0
# Of the ways a possession can end, a turnover in open play is commoner than the whistle -- about
# 110 stoppages against about 300 possessions, so roughly seven ends in ten are turnovers.
TURNOVER_SHARE = 0.7

# No possession shorter than this is a possession. This is not a smoothing constant but the
# challenge's own definition of a change of possession, which requires *controlled* possession and
# explicitly excludes "a simple touch of the ball without control", "a deflection of a pass that
# does not intercept it" and "an aerial duel won without subsequent control" -- with "a player
# keeps the ball for a few seconds" offered as the proxy. A geometric duration prior cannot
# express that on its own: its mode is one frame, so it tolerates exactly the sub-second blips the
# rules say are not possessions. Enforced after the decode, where it is a stated rule rather than
# a penalty competing with the evidence.
MIN_SEGMENT_SECONDS = 2.0

# Control needs the ball close. This is the length scale past which "closer" stops meaning
# anything: beyond a few metres a player is not in contention, and further distance adds no
# information -- which is why the geometric term saturates rather than growing without bound.
CONTROL_METRES = 4.0
# Weight on the saturated geometric cue, in log-odds at saturation. The nearest-track distance is
# the noisiest quantity in the export, so this stops deliberately short of certainty.
GEOMETRY_GAIN = 2.2

# A ball off the pitch is out of play by the Laws of the Game, so this is near-certainty rather
# than a fitted weight. It is capped rather than infinite because the observation under it is an
# automatic detection, and those are occasionally wrong.
OUT_OF_PLAY_MARGIN = 0.002
OUT_OF_BOUNDS_LOGODDS = 3.0

# Base rate of the ball being out of play, as log-odds, and the one parameter the dead/alive call
# is most sensitive to. Set from external football knowledge rather than from this match: ball-in-
# play time at this level runs a little under an hour against a wall clock of around 100 minutes
# once stoppage time is included, so a little under half of the graded frames are dead. It is a
# base rate and not a target -- the evidence below moves it frame by frame.
DEAD_SHARE_PRIOR = 0.45

# The tracker loses the ball far more often while play is stopped than while it is running, but a
# missing ball is close to a coin flip on its own, so it enters as a nudge. Reading it as a
# dead-ball label is the reference baseline's main weakness -- its D class is exactly "no ball".
BALL_MISSING_LOGODDS = 0.55
# A restart window brackets the stoppage that preceded it, to about a second, so it raises the
# dead-ball prior without deciding anything.
STOPPAGE_WINDOW_LOGODDS = 0.5

# Smart Tagging is inferred, not ground truth, so one vote is worth rather less than the geometry
# and several agreeing votes are worth more. Saturating keeps a pile-up of mirror tags from
# overwhelming a ball the tracker can actually see.
TAGGING_GAIN = 1.3
TAGGING_SATURATION = 2.0

# -- parameters: unsupervised -------------------------------------------------------------------
# Collective player speed is the one cue present on every frame, but its distribution is not a
# clean mixture of dead and alive: fitting two components to it by EM splits low-intensity play
# from high-intensity play, and plenty of live football is slow. Fitted that way the low component
# came out at 1.15 m/s with sigma 0.43 and claimed 53% of the match, which is the wrong question
# answered confidently -- a narrow component wins a likelihood ratio near its own mean whatever
# the truth is.
#
# So the two classes are pinned by physics instead, on frames where the state is not in doubt:
#
#   certainly dead    the ball is tracked and off the pitch. Out of play by the Laws of the Game,
#                     and P(out of bounds | alive) is 0.000 over the published window.
#   certainly alive   the ball is tracked, inside the pitch and travelling faster than a ball
#                     ever travels during a stoppage.
#
# Those are pseudo-labels produced by an algorithm with no person deciding any of them, which is
# what RULES.md permits on the provided game. They locate and scale the two speed distributions
# and are never used as labels in their own right.
MIN_SEED_FRAMES = 1000
SPEED_HIST_MAX = 8.0
SPEED_HIST_STEP = 0.1
SPEED_LOGODDS_CAP = 2.5

# Bisection steps for the dead-ball calibration. Each step is a full decode of both halves, so
# this is the run time; 18 steps resolve the offset to under 0.001 nats, well past the point where
# another step changes a single label. Fixed rather than tolerance-based so the output is
# bit-identical on every run, which the reproduction requirement in RULES.md asks for.
CALIBRATION_STEPS = 18


def graded_mask(game=GAME, n=None):
    """Boolean over video frames, True on the two halves. Everything else is never scored."""
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def tagging_signals(n, game=GAME, data_dir=None):
    """(team_vote, stoppage_window) over video frames.

    team_vote is (n, 2) counts of phases pointing at each side; stoppage_window is boolean where a
    set piece or a goal is tagged.
    """
    vote = np.zeros((n, 2), dtype=np.int16)
    stoppage = np.zeros(n, dtype=bool)

    for event in load_tagging(game, data_dir):
        start = max(0, event["start_frame"])
        stop = min(n, event["end_frame"] + 1)
        if stop <= start:
            continue
        code = (event["code"] or "").strip().upper()
        if code in STOPPAGE_CODES:
            stoppage[start:stop] = True
            continue
        # Throw-ins and 35 other instances carry Team = N/A, so a missing side is normal, and it
        # is simply no vote rather than a reason to guess one.
        team = TEAM_CODE.get((event["tags"].get("Team") or "").strip())
        if team is None:
            continue
        if code in OFFENSIVE:
            vote[start:stop, team] += 1
        elif code in DEFENSIVE:
            vote[start:stop, 1 - team] += 1
    return vote, stoppage


def off_pitch(table):
    """Boolean: the ball is tracked and outside the pitch rectangle.

    The margin absorbs the touchline itself, where a ball on the line is still in play and the
    detection is only accurate to a few centimetres of normalised pitch.
    """
    margin = OUT_OF_PLAY_MARGIN
    return table["ball_seen"] & (
        (table["ball_x"] < -margin) | (table["ball_x"] > 1.0 + margin)
        | (table["ball_y"] < -margin) | (table["ball_y"] > 1.0 + margin))


def dead_seed(table, game=GAME):
    """Frames the Laws of the Game put out of play, with no label and no threshold involved.

    A ball off the pitch is dead, and it stays dead until it is back on the pitch -- so the seed
    is not the crossing itself but the stretch that follows it, up to the next frame where the
    ball is tracked inside the pitch again. That extension is what makes the seed usable: the
    crossing alone catches only the moment play stops, while players are still running from the
    action before it, and misses the settled part of the stoppage entirely. Extending it moves the
    seed's median collective speed from 1.25 m/s to 1.09 and quadruples its size.

    Carried forward within a half and never across one, so halftime cannot leak a state into the
    second half.
    """
    n = len(table["ball_x"])
    out_of_play = off_pitch(table)
    back_in_play = table["ball_seen"] & ~out_of_play

    event = np.zeros(n, dtype=np.int8)
    event[back_in_play] = -1
    event[out_of_play] = 1  # a crossing wins over a sighting on the same frame

    seed = np.zeros(n, dtype=bool)
    index = np.arange(n)
    for _, first, last in half_video_blocks(game):
        window = slice(first, last + 1)
        marked = np.where(event[window] != 0, index[window], -1)
        carry = np.maximum.accumulate(marked)
        known = carry >= 0
        seed[window] = np.where(known, event[np.clip(carry, 0, n - 1)] == 1, False)
    return seed


def fit_speed_model(table, graded, game=GAME):
    """Gaussian speed model for the dead and alive classes, in that order.

    Only the dead class is observed directly. The alive class is recovered by deconvolution: the
    speed distribution over all graded frames is a mixture of the two, so subtracting the dead
    component at the stated base rate leaves the alive one,

        p(speed) = pi * p(speed | dead) + (1 - pi) * p(speed | alive)

    which is worth the twenty lines because every attempt to *seed* the alive class is biased.
    Selecting frames where the ball is travelling fast selects moments when players are sprinting,
    and gives an alive mean of 2.39 m/s against a true one nearer 2.2; the model then reads the
    median frame of the match as dead and labels 73% of it a stoppage. Deconvolution needs no
    alive seed at all, so there is no alive-side bias to correct.

    Returned as moments rather than as the histograms they are computed from, because the
    histogram log-ratio is non-monotonic at the sparse low-speed end -- it would say a motionless
    pitch is less likely to be dead than a walking one, which is not true of football.
    """
    speed = table["speed_mean"]
    known = np.isfinite(speed) & graded
    seed = dead_seed(table, game) & known
    if seed.sum() < MIN_SEED_FRAMES:
        raise ValueError("dead seed too small: {} frames".format(int(seed.sum())))

    edges = np.arange(0.0, SPEED_HIST_MAX + SPEED_HIST_STEP, SPEED_HIST_STEP)
    centres = 0.5 * (edges[:-1] + edges[1:])

    def density(mask):
        values = np.clip(speed[mask], 0.0, SPEED_HIST_MAX - 1e-6)
        counts, _ = np.histogram(values, edges)
        # Laplace smoothing: an empty bin must not become a log of zero.
        return (counts + 1.0) / (counts.sum() + len(counts))

    pi = DEAD_SHARE_PRIOR
    marginal = density(known)
    dead = density(seed)
    alive = np.clip(marginal - pi * dead, 1e-9, None)
    alive /= alive.sum()

    def moments(weights):
        mean = float((weights * centres).sum())
        variance = float((weights * (centres - mean) ** 2).sum())
        return mean, max(np.sqrt(variance), 1e-3)

    dead_mu, dead_sigma = moments(dead)
    alive_mu, alive_sigma = moments(alive)
    return {"mu": np.array([dead_mu, alive_mu]),
            "sigma": np.array([dead_sigma, alive_sigma]),
            "n_seed": int(seed.sum())}


def speed_logodds(speed, model):
    """log p(speed | dead) - log p(speed | alive), and 0 where the speed is unknown.

    Capped, so one extreme frame cannot outvote the duration prior on its own. The cap is what
    keeps this a cue rather than a decision.
    """
    out = np.zeros(len(speed), dtype=np.float64)
    known = np.isfinite(speed)
    x = speed[known].astype(np.float64)

    mu, sigma = model["mu"], model["sigma"]
    dead = -0.5 * ((x - mu[0]) / sigma[0]) ** 2 - np.log(sigma[0])
    alive = -0.5 * ((x - mu[1]) / sigma[1]) ** 2 - np.log(sigma[1])
    out[known] = np.clip(dead - alive, -SPEED_LOGODDS_CAP, SPEED_LOGODDS_CAP)
    return out


def enforce_min_duration(codes, min_frames):
    """Absorb every segment shorter than min_frames into a neighbour, shortest first.

    This is where CHALLENGE.md's control requirement is applied: a run too short to be controlled
    possession is a touch, a deflection or a duel, and the rules say none of those is a change of
    possession. Each offending run is given to its longer neighbour, which is the choice that
    preserves the surrounding structure rather than inventing a third state; doing the shortest
    first means a blip never decides the fate of a longer run next to it.

    Works on the segment list rather than the frame array, so the cost is in the number of
    segments and not in the 80000 frames of a half.
    """
    if min_frames <= 1 or not len(codes):
        return codes

    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(codes)]))
    runs = [[int(bounds[i]), int(bounds[i + 1]), int(codes[bounds[i]])]
            for i in range(len(bounds) - 1)]

    while len(runs) > 1:
        lengths = [stop - start for start, stop, _ in runs]
        shortest = min(range(len(runs)), key=lambda i: lengths[i])
        if lengths[shortest] >= min_frames:
            break

        if shortest == 0:
            winner = 1
        elif shortest == len(runs) - 1:
            winner = shortest - 1
        else:
            before, after = lengths[shortest - 1], lengths[shortest + 1]
            winner = shortest - 1 if before >= after else shortest + 1
        runs[shortest][2] = runs[winner][2]

        # Coalesce: the absorbed run now matches a neighbour, and possibly bridges two runs of the
        # same code into one, which is the merge that actually lengthens a possession.
        merged = [runs[0]]
        for run in runs[1:]:
            if run[2] == merged[-1][2]:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        runs = merged

    out = codes.copy()
    for start, stop, code in runs:
        out[start:stop] = code
    return out


def evidence(table, vote, stoppage, model):
    """(dead_logodds, team_logodds) per frame, before any calibration.

    Two independent questions, kept apart so each stays interpretable and so the team evidence can
    be absent -- which it is on 41% of graded frames -- without disturbing the alive/dead call.
    dead_logodds is uncalibrated: it ranks frames by how dead they look, and calibrate_offset
    below fixes the level.
    """
    n = len(table["ball_x"])

    dead = speed_logodds(table["speed_mean"], model)
    dead += np.where(table["ball_seen"], 0.0, BALL_MISSING_LOGODDS)
    dead += np.where(stoppage, STOPPAGE_WINDOW_LOGODDS, 0.0)
    dead += np.where(off_pitch(table), OUT_OF_BOUNDS_LOGODDS, 0.0)

    # Positive favours home. The geometric term saturates with tanh: beyond the control radius a
    # player is simply not in contention, and further distance says nothing more.
    d_home, d_away = table["d1_home"], table["d1_away"]
    usable = np.isfinite(d_home) & np.isfinite(d_away)
    team = np.zeros(n, dtype=np.float64)
    team[usable] = GEOMETRY_GAIN * np.tanh(
        (d_away[usable] - d_home[usable]) / CONTROL_METRES)
    team += TAGGING_GAIN * np.tanh(
        (vote[:, H].astype(np.float64) - vote[:, A]) / TAGGING_SATURATION)
    return dead, team


def assemble(dead, team, offset):
    """(n, 3) emission log-probabilities, columns H / A / D, with `offset` added to the dead score.

    Every column is a log-probability and exp(row) sums to one. That normalisation is not
    cosmetic: a raw log-odds in the D column beside log-sigmoids in H and A is two different
    quantities in one matrix, and the unbounded one wins every comparison whatever the evidence
    says -- which is exactly what happened before it was fixed, labelling 61% of the match dead.
    """
    shifted = dead + offset
    log_dead = -np.logaddexp(0.0, -shifted)
    log_alive = -np.logaddexp(0.0, shifted)

    out = np.empty((len(dead), 3), dtype=np.float64)
    out[:, H] = log_alive - np.logaddexp(0.0, -team)
    out[:, A] = log_alive - np.logaddexp(0.0, team)
    out[:, D] = log_dead
    return out


def decode(dead, team, offset, log_transition, min_frames, game=GAME):
    """Decoded state codes over the whole video, -1 outside the graded halves.

    Each half is decoded on its own, so no path and no possession can run through halftime.
    """
    emit = assemble(dead, team, offset)
    codes = np.full(len(dead), -1, dtype=np.int8)
    for _, first, last in half_video_blocks(game):
        window = slice(first, last + 1)
        codes[window] = enforce_min_duration(
            viterbi(emit[window], log_transition), min_frames)
    return codes


def calibrate_offset(dead, team, log_transition, min_frames, graded, target, game=GAME):
    """The offset for which the *decoded* dead share over graded frames equals `target`.

    Two things make this necessary rather than decorative.

    The dead-ball cues are not independent -- a stoppage is simultaneously slow, missing its ball
    and inside a tagged restart window -- so adding their log-odds counts one stoppage three times
    and the sum runs hot. Adding them is still the right way to *rank* frames, since each cue does
    carry information about which frames are deadest; it is only the absolute level that the
    double counting destroys.

    And the level has to be fixed on the decode rather than on the emission, because the duration
    prior moves it: calibrating the emission's mean posterior to 45% still decoded to 56% dead,
    since D is the stickiest state and Viterbi spends longer there than the per-frame evidence
    alone implies. Calibrating what actually comes out closes that gap, and the quantity being
    enforced is external football knowledge rather than anything measured on this match.

    A useful side effect: every weight above becomes scale-robust. Multiplying all the cues by a
    constant is absorbed here, so what matters is their size relative to each other and not their
    absolute values -- the part a hand-set weight is least likely to get right.

    Decoded dead share is monotone non-decreasing in the offset, so bisection converges. It is run
    to a fixed iteration count rather than a tolerance so the result is bit-identical on every run.
    """
    low, high = -12.0, 12.0
    total = float(graded.sum())
    for _ in range(CALIBRATION_STEPS):
        middle = 0.5 * (low + high)
        codes = decode(dead, team, middle, log_transition, min_frames, game)
        share = float(np.sum((codes == D) & graded)) / total
        if share < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def transition_matrix():
    """(3, 3) log transition probabilities, from the duration priors at the top of the file.

    A state is left at the rate implied by its mean length, and the ways of leaving it are split
    by how a possession usually ends. This is the only place the model is told that possessions
    and stoppages are long, and it is what makes the decoded segments whole.
    """
    leave_possession = 1.0 / (POSSESSION_SECONDS * FPS)
    leave_dead = 1.0 / (DEAD_SECONDS * FPS)

    matrix = np.zeros((3, 3), dtype=np.float64)
    for side, other in ((H, A), (A, H)):
        matrix[side, side] = 1.0 - leave_possession
        matrix[side, other] = leave_possession * TURNOVER_SHARE
        matrix[side, D] = leave_possession * (1.0 - TURNOVER_SHARE)
    matrix[D, D] = 1.0 - leave_dead
    # A restart can go to either side, and which one is not inferable from a three-state chain, so
    # it is left even rather than guessed and the emission at the restart decides it.
    matrix[D, H] = matrix[D, A] = leave_dead / 2.0
    return np.log(matrix)


def viterbi(emit, log_transition):
    """Most likely state sequence through one contiguous block of frames.

    Loops over frames because the recursion is sequential, but each step is a 3x3 numpy operation,
    so a half decodes in well under a second. The backpointer pass is what separates this from a
    per-frame argmax: a frame is labelled by the best *path* through it, which is why the output
    has whole segments rather than flicker.
    """
    n, k = emit.shape
    score = emit[0].copy()
    back = np.zeros((n, k), dtype=np.int8)

    for i in range(1, n):
        candidate = score[:, None] + log_transition
        best = candidate.argmax(axis=0)
        back[i] = best
        score = candidate[best, np.arange(k)] + emit[i]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def solve(game=GAME, data_dir=None, rebuild=False, verbose=True):
    """Per-frame labels for the whole video. X outside the two graded halves."""
    table = F.load(game, data_dir, rebuild=rebuild)
    n = len(table["ball_x"])
    graded = graded_mask(game, n)

    vote, stoppage = tagging_signals(n, game, data_dir)
    model = fit_speed_model(table, graded, game)
    if verbose:
        print("speed model: dead seeded by the Laws of the Game, alive by deconvolution")
        print("  dead seed {} frames ({:.1%} of graded)".format(
            model["n_seed"], model["n_seed"] / float(graded.sum())))
        for index, name in ((0, "dead"), (1, "alive")):
            print("  {:<6} mu {:.3f} m/s   sigma {:.3f}".format(
                name, model["mu"][index], model["sigma"][index]))

    dead, team = evidence(table, vote, stoppage, model)
    log_transition = transition_matrix()
    min_frames = int(round(MIN_SEGMENT_SECONDS * FPS))

    offset = calibrate_offset(dead, team, log_transition, min_frames, graded,
                              DEAD_SHARE_PRIOR, game)
    codes = decode(dead, team, offset, log_transition, min_frames, game)
    if verbose:
        print("dead-ball calibration: offset {:+.3f} nats -> decoded dead share {:.1%}".format(
            offset, float(np.sum((codes == D) & graded)) / float(graded.sum())))

    labels = np.full(n, "X", dtype="<U1")
    inside = codes >= 0
    labels[inside] = LABELS[codes[inside]]

    # Each half is graded from its first frame, but the second does not kick off until 113742 and
    # those frames are not play. Both numbers are published in this repository as boundaries --
    # FAQ.md and starter/load_smart_tagging.py -- so this is given geometry and not anything read
    # off a label.
    for half, first, _ in half_video_blocks(game):
        kickoff = KICKOFF_VIDEO_FRAMES.get("{} Half".format("1st" if half == 1 else "2nd"))
        if kickoff is not None and kickoff > first:
            labels[first:kickoff] = "D"
    return labels


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def describe(labels, game=GAME):
    """Possession count, mean duration and home share -- what the ranking metric actually sees.

    Counted with the scorer's own possession_spells rather than a second implementation of "a
    maximal run of H or of A", so these numbers cannot drift from the ones the board reports.
    """
    sys.path.insert(0, os.path.join(HERE, "evaluation"))
    import possession_metrics as pm

    runs = [(first, last + 1) for _, first, last in half_video_blocks(game)]
    codes = pm.label_codes(labels)
    spells = pm.possession_spells(codes, runs)
    lengths = [stop - start for start, stop, _ in spells]

    graded = graded_mask(game, len(labels))
    counts = {label: int(np.sum(labels[graded] == label)) for label in ("H", "A", "D", "X")}
    alive = counts["H"] + counts["A"]
    print("graded frames  H {H}  A {A}  D {D}  X {X}   dead share {d:.1%}".format(
        d=counts["D"] / float(graded.sum()), **counts))
    if lengths:
        print("possessions {}  mean duration {:.1f} s  home share {:.1%}".format(
            len(lengths), float(np.mean(lengths)) / FPS,
            counts["H"] / float(alive) if alive else 0.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT, help="where to write the submission")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the feature cache first")
    args = parser.parse_args()

    labels = solve(rebuild=args.rebuild)
    print()
    describe(labels)
    print("\nwrote {}".format(os.path.relpath(write_submission(args.out, labels), HERE)))
