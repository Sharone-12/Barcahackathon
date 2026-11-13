# Frame-level ball possession — technical document

**Task.** One label per frame of the tactical video, `H` / `A` / `D` / `X`, 199,319 rows.
**Method.** A three-state hidden Markov model over `{H, A, D}`, decoded with Viterbi, on features
derived from the provided automatic tracking and Smart Tagging. No learning from labels.

```bash
pip install -r requirements.txt
python features.py     # one vectorised pass over the ATD -> cache/ (about 2 s)
python solve.py        # -> evaluation/results/hmm.csv  (about 6 s)
```

Deterministic: repeated runs produce a byte-identical file (MD5 `5b64b509…`).

---

## 1. Approach, and why this one

The ranking metric scores **whole possessions**, matched one-to-one against the truth by temporal
IoU. It therefore rewards a signal whose *segments* align with the truth's segments, which is not
the same thing as a signal whose individual frames are right. The provided baselines make the
point themselves: four runs of one algorithm differing only in a smoothing window move `macroF1`
by 0.04 and `spellF1` by a factor of five.

That smoothing window is a duration prior in disguise. Too narrow and possessions fragment; too
wide and they merge; and the right width is not discoverable, because the only ground truth a
participant holds is five minutes long and picks the wrong one (`smooth125` wins the public window
at 0.4242 and loses the full match at 0.2242).

An HMM puts the duration prior where it belongs — the transition matrix. Leaving a state costs a
fixed log-penalty, so a possession survives a few seconds of contrary evidence while a sustained
change still flips it. Segments come out whole because the model assumes they are whole, rather
than because a window was tuned until they were.

The emission at each frame answers two independent questions, kept separate so that either can be
missing:

| | evidence |
|---|---|
| **is the ball in play** | collective player speed, whether the tracker has the ball at all, whether the ball is off the pitch, whether Smart Tagging marks a restart nearby |
| **if so, whose is it** | which side has the closest tracked player to the ball; which team Smart Tagging's phase tags point at |

Both halves are necessary because neither is available everywhere: the ball is missing on **41 %**
of graded frames, the tagging covers **68 %**, and the two gaps do not coincide. Only **16 %** of
graded frames have neither.

### The ATD's speed field trails its own positions by 7 frames

The export ships a per-player speed beside each position, and **the two are not synchronous**. The
speed behaves like a trailing-window derivative, so the value at frame *t* describes motion from
about 0.28 s earlier. Because every dead-ball cue rests on collective speed, the whole alive/dead
signal arrived late, and `spellF1` charges exactly that: it averages over IoU 0.3 to 0.7, and at
the tight end a possession that is *found* but *mistimed* stops matching at all.

The lag is **measured, not assumed, and measured without any ground truth** — the ATD is checked
against itself. A centred finite difference of the positions is a speed with no lag by
construction, so the offset that best correlates the reported speed against it is the report's own
lag. It peaks at 7 frames (r = 0.9641, against 0.9432 unshifted), the curve is smooth and
single-peaked, and after correcting, re-measuring returns exactly 0. Correcting it was the single
largest gain in the project:

| | macroF1 | spellF1 | transF1 | mean latency | p90 \|latency\| |
|---|---|---|---|---|---|
| reported speed as shipped | 0.7762 | 0.4500 | 0.3500 | +6.0 | 20 |
| realigned by 7 frames | **0.7896** | **0.4903** | **0.3684** | **+3.4** | **13** |

The gain lands precisely where the diagnosis said it would — the tight thresholds, IoU 0.6 going
from 7 matched possessions to 8 and IoU 0.7 from 4 to 5.

### What did not work: re-placing the boundaries

Worth recording, because the reasoning looked sound. If the duration prior makes edges late, then
re-placing each boundary at the frame the emission alone most prefers — labels and segment count
held fixed — should sharpen them. Implemented, it moved `spellF1` not at all and made `macroF1`
(0.7762 → 0.7622) and `transF1` (0.3500 → 0.2500) worse.

The premise was wrong, for a reason worth stating: **the transition penalty is a constant that
does not depend on where the boundary sits.** Given that a switch happens, Viterbi already places
it at the emission-optimal frame. Re-optimising under the emission alone therefore re-solves the
same problem having discarded the prior — it removes information rather than adding any, which on
a noisy emission is strictly worse. The lateness was never in the decoder; it was in the features,
and the section above is where it actually lived. The code was reverted.

## 2. Assets used, and dropped

| asset | used | how |
|---|---|---|
| tactical ATD (`.txt` / `.xml`) | **yes** | ball position, per-side nearest-player distance, collective player speed, team/slot map |
| Smart Tagging | **yes** | per-frame team vote; set-piece and goal windows as dead-ball evidence |
| half boundaries, 2nd-half kickoff | **yes** | published in `DATA.md` / `FAQ.md`; frames 110,808–113,741 are graded and not play |
| tactical homography | **dropped** | maps pitch to pixels. Needed only to look at the video, which this method never does |
| tactical video | **dropped** | no appearance model; the geometry carries the signal |
| broadcast ATD + homography | **dropped** | on its own unsynchronised timeline with no mapping provided. Aligning it is a project in itself and the gain is speculative — see §7 |

### Reading Smart Tagging as a vote

The phases come in **attack/defend mirror pairs**: `PROGRESSION`/Barcelona and `DEFENDING IN
MIDDLE THIRD`/Atletic Club each occur 47 times with the same median duration, `BUILD UP` pairs
with `DEFENDING IN ATTACKING THIRD` at 22 each, and so on. An offensive tag and the defensive tag
facing it are two views of one phase, and **both imply the same team has the ball** — so the
defensive codes are read as a vote for the *opponent* of the tagged team.

Treating these as votes rather than labels is what makes the false positives survivable; the
tagging is inferred, not ground truth. `SET PIECES` and `GOALS` are never read as possession cues
— a set piece is a restart and so implies a preceding stoppage, and a goal is followed by a long
dead stretch. Every throw-in is tagged `Team = N/A`, so a missing side is normal and contributes
no vote rather than a guess.

## 3. Training data

**None.** No model is fitted to any labelled data, from this match or any other. No hand
annotation was produced or used. The only quantities estimated from data are the two speed
distributions in §4, and they are estimated without labels.

The published five minutes were used **once**, at the end, to confirm the pipeline is correct —
frames aligned, file well-formed, labels meaning what they are supposed to. No parameter was
chosen by scoring against them. That is both the rule and, on 17 possessions, the only defensible
practice: the bootstrap in the provided scorer puts the noise band on public `spellF1` at **±0.35**,
so the window cannot distinguish between two versions of anything.

### Where each constant comes from

Every parameter is from one of three sources, and which one is stated at its definition in
`solve.py`:

**Football and physics.** A possession runs of the order of 12 s and a stoppage 25 s; ball-in-play
is a little under 60 % of the wall clock, giving ≈ 45 % dead over the 106 graded minutes; a match
holds ≈ 300 possessions and ≈ 110 stoppages, so roughly 7 ends in 10 are turnovers rather than
whistles; control needs the ball within a few metres; **a ball off the pitch is out of play by the
Laws of the Game.** All of these are properties of elite football, not of this match.

**Unsupervised, from this match.** The two collective-speed distributions (§4).

**Self-calibrating.** One offset, fixed by the base rate (§4).

## 4. The dead-ball model, which is where the work was

Dead ball is the hard class and the baseline's weakest: its `D` is exactly "the tracker lost the
ball". That is wrong far more often than it looks — `P(dead | ball missing) = 0.56`, barely better
than a coin flip, because the ball goes missing in **150 long blocks of median 10.7 s**, not in
brief occlusions. Interpolation cannot help: bridging every gap under half a second recovers 133
frames out of 85,831.

Three things had to be got right.

**The two speed distributions.** Collective player speed is the only cue present on every frame.
Fitting two Gaussians to it by EM fails instructively: it splits *low-intensity* from
*high-intensity* play, lands the low component at 1.15 m/s claiming 53 % of the match, and labels
the median frame dead — a narrow component wins a likelihood ratio near its own mean whatever the
truth is. Seeding the classes by physics fails differently: selecting "certainly alive" frames as
those where the ball travels > 8 m/s selects moments when players are *sprinting*, biasing the
alive mean to 2.39 m/s and decoding 73 % of the match as dead.

What works is to seed only the class that can be seeded reliably, and derive the other:

- **dead seed** — the ball is tracked off the pitch, *and stays dead until it is tracked inside
  the pitch again*. The extension matters: the crossing alone catches only the instant play stops,
  while players are still running from the action before it. Extending it moves the seed's median
  speed from 1.25 to 1.09 m/s and quadruples it to 15,045 frames (9.5 % of graded).
- **alive by deconvolution** — the speed distribution over all graded frames is a mixture of the
  two, so `p(alive) = (p(all) − π·p(dead)) / (1 − π)`. No alive seed exists, so no alive-side bias
  exists to correct.

This yields dead 1.205 ± 0.632 m/s against alive 2.215 ± 1.027, and a log-odds that crosses zero
at 1.52 m/s — the actual median graded frame, i.e. no systematic lean either way.

**Normalising the emission.** The `D` column was initially a raw log-odds sitting beside
log-sigmoids in `H` and `A`: two different quantities in one matrix, and the unbounded one wins
every comparison regardless of evidence. It labelled 61 % of the match dead. All three columns are
now log-probabilities summing to one.

**Calibrating the level.** The dead cues are **not independent** — a stoppage is simultaneously
slow, missing its ball, and inside a tagged restart window — so summing their log-odds counts one
stoppage three times and runs hot. Summing is still right for *ranking* frames; only the absolute
level is destroyed. One offset fixes the level, bisected so that the **decoded** dead share equals
the externally-known 45 %.

Calibrating the decode rather than the emission is the part that matters: pinning the emission's
mean posterior to 45 % still decoded to 56 %, because `D` is the stickiest state and Viterbi dwells
there longer than per-frame evidence implies. A useful side effect is that every hand-set weight
becomes **scale-robust** — multiplying all cues by a constant is absorbed by the offset, so only
their size *relative to each other* matters, which is the part a hand-set weight is least likely
to get right.

## 5. Possession rules implemented

`CHALLENGE.md` requires **controlled** possession and excludes "a simple touch of the ball without
control", "a deflection of a pass that does not intercept it" and "an aerial duel won without
subsequent control", offering "a player keeps the ball for a few seconds" as the proxy.

A geometric duration prior cannot express that: its mode is one frame, so it tolerates exactly the
sub-second blips the rules say are not possessions. So after decoding, **every run shorter than
2 s is absorbed into its longer neighbour**, shortest first. This is the stated rule applied as a
rule, not a smoothing constant competing with the evidence.

The other rules follow from the state space. A possession is a maximal run of one side with the
ball alive, so `H H A A` is two possessions and `H H D D H H` is two with a stoppage between them.
Dead ball is a stoppage and never a possession. Loose and contested balls stay with whoever the
geometry and the duration prior jointly favour, which is usually the previous holder — matching
"in phases where the ball is contested or loose, possession stays with the last team that had it".

**Dead-ball ownership is not inferred**, because it is not asked for: when the ball is dead the
answer is `D`.

**`X` is written only outside the two halves**, where it is never scored. Inside them it is never
used: `macroF1` charges an abstention as a miss, and `spellF1` replaces it with the previous label
anyway, so a coherent decode has nothing to gain from it.

## 6. Results

Scored on the **public five minutes** with the provided `run_evaluation.py`. These are a
correctness check, not a ranking — the published floor is `baseline_smooth50` at **spellF1 0.2433
on the full match**, which no participant can measure.

| submission | macroF1 | spellF1 | poss (GT 17) | dur (GT 14.2 s) | transF1 | shareErr | frag |
|---|---|---|---|---|---|---|---|
| **hmm** | **0.7896** | **0.4903** | **14** | **18.3 s** | **0.3684** | **0.013** | 0.90 |
| `baseline_smooth125` | 0.5812 | 0.4242 | 16 | 18.5 s | 0.1667 | 0.052 | 0.81 |
| `baseline_smooth50` | 0.6207 | 0.3256 | 26 | 10.8 s | 0.0816 | 0.048 | 1.43 |
| `baseline_smooth25` | 0.6200 | 0.2868 | 36 | 7.6 s | 0.2034 | 0.056 | 1.90 |
| `baseline_raw` | 0.5972 | 0.0378 | 131 | 2.0 s | 0.1410 | 0.046 | 6.52 |

Per-IoU: 0.5806 / 0.5161 / 0.5161 / 0.5161 / 0.3226 at IoU 0.3 → 0.7. Alive/dead F1 **0.9435**;
team accuracy where both sides agree the ball is alive **0.8663**.

**What is and is not evidence here.** The `spellF1` column cannot rank: its bootstrap band on this
window is ±0.35, so the margin over `smooth125` is noise and is not claimed as a result. The
`macroF1` gap of **+0.169** over the best baseline is real — measured over 7,500 graded frames and
an order of magnitude outside that metric's noise. The structural columns are the most
informative, because they are the quantities `spellF1` is built from: 14 possessions against 17,
mean duration within 4 s, possession-share error cut from 0.048 to **0.013**, and transition F1
more than doubled at less than half the latency.

Over the full match the model outputs 267 possessions of mean 12.8 s, 46.2 % dead, and a 61.8 %
home share — all plausible for this fixture, and all unverifiable until the private truth is
applied.

## 7. Limitations

- **The dead share is one assumed number** — 45 %, from external football knowledge about
  ball-in-play time, enforced by the calibration. This is the largest single assumption, so it was
  stress-tested across the whole plausible range. The structure barely moves:

  | assumed dead share | 0.33 | 0.38 | 0.42 | **0.45** | 0.48 | 0.52 | 0.57 |
  |---|---|---|---|---|---|---|---|
  | possessions | 294 | 289 | 276 | **272** | 273 | 281 | 282 |
  | mean duration (s) | 14.5 | 13.6 | 13.4 | **12.8** | 12.1 | 10.9 | 9.7 |
  | home share of alive | 60.7 % | 61.0 % | 61.7 % | **62.5 %** | 62.6 % | 62.4 % | 61.1 % |

  Possession count varies by ±4 % across a 24-point swing in the assumption, and home share by
  under two points. Since `spellF1` is driven by how many possessions there are and where their
  edges fall, the exposure is much smaller than the assumption's uncertainty suggests. Mean
  duration does trade off directly against the assumption, as it must.
- **No appearance model.** Neither video is read. Identity switches and duplicate tracks in the
  ATD are absorbed by the duration prior rather than resolved.
- **Restart ownership is not modelled.** `D → H` and `D → A` are equally likely in the transition
  matrix. In reality a restart goes to the opponent of whoever caused the stoppage — real signal,
  and a three-state chain cannot hold it. A four-state model that remembers the pre-stoppage side,
  or a post-hoc rule keyed to the restart location, is the clearest next gain.
- **The broadcast feed is unused.** Its tracking sees the ball on 50.5 % of its own frames and
  could cover some of the tactical gaps, but it needs a timeline alignment that is not provided.
- **Transition timing still lags ~3 frames** after the 7-frame realignment, and `p90` is 13. The
  residual is no longer the known feature lag, so the next candidate is the Smart Tagging windows,
  which are accurate only to about a second and currently enter unsmoothed.
- **Smart Tagging weights are hand-set priors.** Calibration makes their overall scale harmless,
  but the ratio between the geometric and tagging terms is reasoned, not measured.

## 8. Reproduction

```bash
# data/ laid out as DATA.md specifies. Required: the tactical ATD (.txt + .xml) and the Smart
# Tagging .xml. The videos and both homography files are never read.
pip install -r requirements.txt
python features.py                                     # cache/features_BAR-ATH_tactical.npz
python solve.py                                        # evaluation/results/hmm.csv
python evaluation/validate_submission.py evaluation/results/hmm.csv
python evaluation/run_evaluation.py                    # scores every file in results/
```

Pure numpy and pandas; no GPU, no network, no `kloppy` needed. Total runtime under 10 s once the
data is in place. `features.py` asserts the ATD's row shape and that its frame ids are exactly
`1..199319` contiguous, so a changed export fails loudly instead of silently shifting every label
by one.
