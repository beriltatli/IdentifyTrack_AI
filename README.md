When an object disappears behind something for two seconds, what does it take to give it back its original identity — and how would you even measure whether you succeeded?

# Reappear

*Identity-preserving multi-object tracking under occlusion.*

A multi-object tracker whose job is **association, not detection**: the detector is frozen and never
touched, and the evaluation is built to separate "is there an object here" (DetA) from "is this the
*same* object I saw 40 frames ago" (AssA).

## TL;DR — the honest result

**The full tracker does not beat a plain Kalman + IoU tracker.** Appearance memory, a lost-track buffer,
gated gallery updates and two-stage matching together made things *slightly worse*, not better, on
MOT17-02/04/09. Three of the success criteria I set were **missed** (table below). DetA is identical
across every variant (55.0), so the comparison is clean, and the failure is informative:
the taxonomy shows the appearance memory traded *memory* failures (object returns with a new ID) for
*drift* failures (an old ID is wrongly handed to someone else).

| | criterion | result |
|---|---|---|
| **MISSED** | HOTA >= 55 | **49.4** (DetA 55.0, AssA 44.5) |
| **MISSED** | long-gap (40+) ID recovery >= 60%, and clearly above Kalman+IoU | **11.3%** (7/62) vs Kalman+IoU **17.7%** (11/62) |
| **MISSED** | ID switches >= 30% below Kalman+IoU at equal DetA | **668 vs 592 — 12.8% *more*** |
| PASS | DetA varies < 1 pt across all variants | spread **0.00** (55.0 everywhere; 74.1 on oracle boxes) |
| PASS | oracle-detection run reported, AssA gap quantified | AssA oracle − real = **+24.8** pts (ours) |
| PASS | own Hungarian equals scipy's total cost on 500 random matrices | 500/500, incl. 461 rectangular, 121 with forbidden pairs |

Because the tracker does not beat Kalman + IoU on the 40+ bin by any margin, **the appearance memory is
not earning its complexity here.** Two further facts limit how much blame the design deserves:

* With `lost_buffer = 100`, at most **36 of the 62** long-gap events are recoverable by construction
  (26 gaps are longer than 100 frames; median long gap = 84 frames ≈ 2.8 s, max 279). 36/62 = 58% is
  below the 60% target, so **the 60% criterion was unreachable in the locked configuration** whatever
  the appearance model did. Of the 36 reachable events the tracker recovered 7.
* In 14 of the 62 long gaps the detector never sees the person again, so nothing can be associated.
  These count as failures (strict definition) but are detection, not association, failures.

## Why MOTA is not the headline

MOTA = 1 − (FN + FP + IDSW) / #GT. It is dominated by false positives and misses — that is, by the
detector. An identity switch costs exactly as much as one missed box, so a tracker can shred every
identity in the scene and still post a respectable MOTA as long as the detector is good. A unit test
(`tests/test_hota.py::test_mota_hides_identity_shredding`) builds exactly this: a perfectly detected
object given a new ID every 10 frames scores **MOTA 0.91**, **AssA 0.10**, HOTA 0.32.

So the headline is **HOTA, always decomposed into DetA and AssA**; IDF1 is reported next to it;
MOTA appears only as a legacy column, marked as detector-dominated. `DetA` is the detector's column
and must not move between trackers; a spread ≥ 1 point is flagged automatically
(`eval.report.deta_warnings`), never averaged away.

## Setup

| | |
|---|---|
| Data | MOT17-02, -04, -09 (static camera; 2175 frames, 171 GT tracks). Moving-camera sequences excluded on purpose: no camera-motion compensation. |
| Detector | **Frozen black box**: MOT17's public SDP detections (`det.txt`), never trained or tuned. The benchmark pre-thresholds them at 0.4 (see limitations). A YOLO backend is implemented (`detection/frozen_detector.py`) but see "YOLO" below. |
| Appearance | Main runs: hand-made colour histogram (3 stripes, chromaticity + brightness, Hellinger-normalised). Extra: ImageNet ResNet18 (512-d). Neither is trained for person re-identification. |
| Occlusion | MOT17 keeps annotating a person who is fully hidden. A GT box with **visibility < 0.25** counts as *hidden*; hidden stretches define the occlusion gaps. Locked before any result was seen. HOTA/IDF1 still use *all* GT boxes. |
| Emission contract | Every tracker outputs **exactly the detector boxes with score ≥ `output_conf` (0.30)**, raw, each carrying an ID. Trackers decide IDs, never which boxes exist — this is what makes DetA identical by construction rather than by luck. |
| Gap definition | A *reappearance* is recovered iff the first matched pred ID after the gap equals the last matched pred ID before it. No prior ID → excluded (14). Never matched again in that visible segment → counted as **failure**. Bins: 1–5, 6–15, 16–39, **40+** (so "40+" literally means ≥ 40). |

Everything is controlled by `config.yaml`; values marked LOCKED were fixed before results were seen.
Whole pipeline is deterministic (I ran the main comparison twice: identical to the last digit).

## Main table (real detections, colour-histogram appearance, 3 sequences pooled)

DetA is constant because all trackers emit the same boxes.

| method | HOTA | **DetA** | AssA | IDF1 | ID switches | long-gap (40+) recovery | MOTA* |
|---|---|---|---|---|---|---|---|
| Greedy IoU | 46.8 | **55.0** | 39.9 | 52.5 | 1593 | 3.2% (2/62) | 60.6 |
| Kalman + IoU (SORT-equivalent) | **52.6** | **55.0** | **50.4** | **63.2** | **592** | **17.7% (11/62)** | 62.0 |
| Appearance only | 43.1 | **55.0** | 33.9 | 48.1 | 1325 | 14.5% (9/62) | 61.0 |
| **Ours (full)** | 49.4 | **55.0** | 44.5 | 56.7 | 668 | 11.3% (7/62) | 61.9 |

\* MOTA is a legacy column, dominated by the detector (see above).

Fragmentation (distinct pred IDs per GT track, median / 90th percentile): greedy 7.0 / 26.3, Kalman+IoU
3.0 / 8.3, appearance-only 3.0 / 9.0, ours 3.0 / 9.0. Tracker-only speed (association only, excluding
detection and feature extraction; pure Python/numpy on one CPU core): greedy ≈ 14 000 fps, Kalman+IoU
≈ 1 200 fps, appearance-only ≈ 1 300 fps, ours ≈ 390 fps. Colour-histogram extraction ≈ 270 fps
including JPEG decode of 1080p frames.

Per sequence (MOT17-09 was the **development** sequence I debugged on, so 02+04 is the less biased number):

| sequence | Kalman+IoU HOTA / AssA / IDSW / 40+ | Ours HOTA / AssA / IDSW / 40+ |
|---|---|---|
| 02 | 30.9 / 27.0 / 372 / 6 of 39 | 30.0 / 25.3 / 363 / 4 of 39 |
| 04 | 60.0 / 56.8 / 172 / 5 of 17 | 56.5 / 50.3 / 262 / 3 of 17 |
| 09 (dev) | 48.2 / 42.7 / 48 / 0 of 6 | 42.1 / 32.5 / 43 / 0 of 6 |
| 02 + 04 | **52.9 / 51.0** / 544 / 11 of 56 | 50.0 / 45.5 / 625 / 7 of 56 |

## The centrepiece: identity recovery by occlusion-gap length

Reappearances of a hidden object, and how many got their **original ID** back (recovered / total).

| method | gap 1–5 | gap 6–15 | gap 16–39 | **gap 40+** |
|---|---|---|---|---|
| Greedy IoU | 16/46 (35%) | 7/63 (11%) | 9/63 (14%) | 2/62 (3%) |
| Kalman + IoU | 24/46 (52%) | 32/63 (51%) | 29/63 (46%) | **11/62 (18%)** |
| Appearance only | 21/46 (46%) | 26/63 (41%) | 24/63 (38%) | 9/62 (15%) |
| **Ours** | 23/46 (50%) | 31/63 (49%) | 28/63 (44%) | 7/62 (11%) |

Two things this table shows that HOTA hides. First, **every method collapses as the gap grows**: the best
tracker recovers about half the identities after a few frames and under a fifth after 40+. A number like
HOTA 52 says nothing about this. Second, the failure is **not** uniform across methods: greedy IoU has
no memory and loses essentially everything beyond 5 frames; Kalman + IoU's motion prediction carries it
through short and medium gaps but not past ~1.5 s.

## ID-switch taxonomy (what kind of mistake is it?)

Every switch is classified automatically, in this order: **swap** (two tracks exchange IDs — a matching
failure), **drift** (an ID that belonged to another object lands on this one, non-mutually — a motion-model
failure), **gap-reassign** (object returned after occlusion with a stranger's new ID — a memory failure),
**other** (continuously visible but its track died and a fresh ID started — a lifetime failure; not one of
the three requested causes, kept separate so counts add up instead of forcing a label). A swap incident
produces two events.

| method | swap | drift | gap-reassign | other | total |
|---|---|---|---|---|---|
| Greedy IoU | 0 | 196 | 527 | 870 | 1593 |
| Kalman + IoU | 12 | 186 | 175 | 219 | 592 |
| Appearance only | 79 | 920 | 16 | 310 | 1325 |
| Ours | 34 | 386 | 97 | 151 | 668 |

Reading it: greedy IoU fails on *memory* (527 gap-reassigns) and lifetime (870). Appearance-only has
almost no memory failures (16) but enormous drift (920) and 79 swaps: it remembers well and matches
badly, because nothing ties an ID to a place. **Ours vs Kalman + IoU**: memory failures drop (175 → 97)
and lifetime failures drop (219 → 151), but drift **doubles** (186 → 386). The appearance memory
turns "came back as a stranger" into "was wrongly given someone else's old identity". With weak
appearance features the second error is as costly as the first.

### Before / after the gallery gating change

| gallery policy | swap | drift | gap-reassign | other | total | HOTA | 40+ recovery |
|---|---|---|---|---|---|---|---|
| always update (naive) | 51 | 395 | 88 | 145 | 679 | 49.2 | 10/62 (16%) |
| **occlusion-gated** | **34** | 386 | 97 | 151 | 668 | 49.4 | 7/62 (11%) |

Gating cut swaps by a third (51 → 34), which is the direction the poisoning argument predicts — but it
barely changed the total (679 → 668), HOTA (+0.2), and long-gap recovery got *worse* (10 → 7).

## Oracle: association with detection removed

Same trackers, but the detections are the ground-truth boxes (objects with visibility < 0.25 withheld —
a detector cannot see through an occluder; without this the tracker would never lose anything and the oracle
would have no gaps to test). AssA is then pure
association quality. DetA is 74.1 for all four oracle runs and 55.0 for all four real runs.

| method | AssA oracle | AssA real | gap | HOTA oracle | IDSW oracle | 40+ oracle |
|---|---|---|---|---|---|---|
| Greedy IoU | 67.8 | 39.9 | +27.9 | 70.9 | 249 | 0/64 |
| Kalman + IoU | 79.8 | 50.4 | +29.3 | 76.9 | 63 | 24/64 (38%) |
| Appearance only | 51.5 | 33.9 | +17.6 | 61.8 | 236 | 6/64 (9%) |
| Ours | 69.3 | 44.5 | **+24.8** | 71.7 | 99 | 20/64 (31%) |

About **25 AssA points of my tracker's error are the detector's fault, not association's** (it is 29 for
Kalman + IoU). But it also says the design is not the detector's victim: *even with perfect boxes*,
Kalman + IoU (AssA 79.8, 38% long-gap) beats the full tracker (69.3, 31%).

## Ablations and sweeps (real detections; locked defaults in bold)

**(a) gallery update policy** — table above. **(e) appearance at all**

| variant | HOTA | AssA | ID switches | 40+ recovery |
|---|---|---|---|---|
| motion only | 48.6 | 43.0 | 806 | 8/62 (13%) |
| **motion + appearance** | 49.4 | 44.5 | 668 | 7/62 (11%) |

**(b) lost-buffer length** (frames a lost track keeps coasting and stays matchable)

| lost_buffer | HOTA | AssA | IDF1 | ID switches | 40+ recovery |
|---|---|---|---|---|---|
| 0 | 47.3 | 40.8 | 53.5 | 1609 | 2/62 (3%) |
| 10 | 50.6 | 46.6 | 59.2 | 832 | 7/62 (11%) |
| 30 | **51.0** | **47.3** | **60.2** | 695 | 10/62 (16%) |
| 60 | 50.4 | 46.3 | 59.1 | 680 | 10/62 (16%) |
| **100** | 49.4 | 44.5 | 56.7 | 668 | 7/62 (11%) |
| 200 | 48.9 | 43.6 | 56.3 | 660 | 6/62 (10%) |

**(c) motion weight** (1 = motion only, 0 = appearance only among feasible pairs)

| lambda_motion | HOTA | AssA | ID switches | 40+ recovery |
|---|---|---|---|---|
| 0.0 | **51.2** | **47.7** | 713 | 9/62 (15%) |
| 0.25 | 50.1 | 45.6 | **617** | 10/62 (16%) |
| **0.5** | 49.4 | 44.5 | 668 | 7/62 (11%) |
| 0.75 | 48.7 | 43.2 | 723 | 9/62 (15%) |
| 1.0 | 48.4 | 42.8 | 800 | 9/62 (15%) |

**(f) appearance features** (app_gate re-calibrated per backend with the same rule: p99 of same-identity cosine distance)

| features | AUC same<diff | app_gate | appearance-only HOTA | ours HOTA | ours 40+ |
|---|---|---|---|---|---|
| colour histogram | 0.83 | 0.30 | 43.1 | 49.4 | 7/62 (11%) |
| ImageNet ResNet18 | 0.87 | 0.38 | 46.6 | 49.8 | 8/62 (13%) |

**Which knob mattered?** In order of HOTA range: **lost buffer** (3.7 pts, and 1609 → ~660 switches —
by far the most consequential), **motion weight** (2.8 pts), appearance on/off (0.8), better ReID
features (0.4 for the full tracker, but 3.5 for appearance-only). **The one that mattered least was
the gallery update policy**: 0.2 HOTA, and it moved the long-gap rate in the *wrong* direction. The
design decision I called the most important in the brief is, empirically, nearly irrelevant in this
setting. Two honest caveats: with MOT17's saturated detector scores (below) the confidence half of the
gate is almost always true, so only the overlap half is doing anything; and with weak features there is
little signal for a clean gallery to protect.

The sweeps also show the locked defaults were not optimal (buffer 30 and lambda 0 each give HOTA ≈ 51).
I did **not** re-lock them after seeing this — choosing the best setting on the evaluation data would be
tuning to the test. Even the best single setting (51.2) is below Kalman + IoU (52.6).

## Qualitative examples

![qualitative](figures/qualitative.png)

`figures/qualitative.png`: the longest successful long-gap re-identification and the three longest
failures of the full tracker (selection rule fixed in advance in `scripts/make_figures.py`, not
cherry-picked). Columns: last matched frame before the gap | mid-gap (object hidden, GT position, dashed) |
first matched frame after | 10 frames later. Border colour and number = predicted ID.

| row | sequence / GT id | hidden | frames shown | outcome |
|---|---|---|---|---|
| success | MOT17-02, GT 9 | 93 frames (243–335) | 242, 289, 337, 347 | **ID 6 → ID 6** |
| failure | MOT17-02, GT 18 | 279 frames (157–435) | 155, 296, 436, 446 | ID 14 → **no box ever again** (detector) |
| failure | MOT17-02, GT 17 | 260 frames (197–456) | 196, 327, 457, 467 | ID 10 → **ID 73** |
| failure | MOT17-04, GT 94 | 246 frames (710–955) | 709, 833, 956, 966 | ID 36 → **ID 26** |

The three failures hid the object for 246–279 frames, beyond the 100-frame buffer, so they were unrecoverable by
construction regardless of appearance (and in the first one the detector never finds the person again); the one
success (93 frames) is inside the buffer. Nothing here shows the tracker *could* have recovered the failures. `figures/MOT17-09_success_gt10.mp4` and `..._failure_gt7.mp4` are side-by-side
videos (greedy IoU vs Kalman + IoU) on oracle detections, white frame = the followed object.

## Limitations

* **Few sequences, one class.** Three static-camera pedestrian sequences; 62 long-gap events. One event
  is 1.6 percentage points, so the 40+ rates are noisy. No confidence intervals were computed.
* **No held-out sequence.** `app_gate`, the Mahalanobis gates and the tracker structure were chosen while
  looking at these sequences' GT (MOT17-09 especially). The `app_gate` rule is fixed and stated, but it
  is still calibrated on the evaluated identities. The design was changed four times during debugging
  on MOT17-09 (Mahalanobis gate → tail-calibrated; IoU fallback for border-clipped boxes; recency
  cascade so coasting tracks cannot steal detections; IoU-only for active tracks). These made the
  tracker better and are documented in `tracker/cost.py` and `config.yaml`; they also mean 09 is not
  a fair test.
* **The 60% criterion was unreachable as configured** (see TL;DR: 58% ceiling at `lost_buffer=100`).
  I did not change the buffer to make it reachable.
* **Detector caveat.** The "frozen detector" is MOT17's public SDP output, not a model I ran. The
  benchmark pre-thresholds it at score 0.4 and scores are saturated (median 1.0). Consequences:
  the low-confidence second matching stage has nothing to match and is **inert** in every reported
  number; `gallery_update_conf` barely filters; `conf_floor`/`output_conf` are vacuous. Also, SDP
  may have been trained on MOT data; I could not verify. DetA (55.0) is bounded by it.
* **Weak appearance models.** Colour histograms and ImageNet features, no ReID fine-tuning (AUC
  0.83 / 0.87). A real person-ReID model is the obvious next experiment and could change the picture.
* **No camera-motion compensation**, hence the static-camera sequences only.
* **Strict gap metric.** "Never re-acquired" counts as failure, including when the detector simply
  never finds the person again (14 of 62 long gaps). A gap is any GT-hidden stretch — a person leaving and
  re-entering the frame looks the same to it. Occlusion is defined by GT visibility < 0.25.
* **Official-toolkit differences.** MOT17's distractor classes are not used to forgive false positives,
  and HOTA here is my own implementation of the paper's definition; absolute numbers are not
  comparable with the leaderboard.
* **Offline evaluation only.** Runtime is not optimised (FPS reported above, association only).
* **Sweeps have a single seed** — the pipeline is deterministic, so there is no seed variance, but
  also no error bars.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy pytest pyyaml      # core (scipy: tests only)
.venv/bin/pip install torch torchvision                                         # only for the ResNet18 experiment
.venv/bin/python -m scripts.fetch_data && .venv/bin/python -m scripts.fetch_dets  # MOT17-02/04/09 via HTTP Range (needs curl)
.venv/bin/python -m scripts.reproduce    # tests -> main table -> all sweeps -> figure -> criteria check
```

`scripts.reproduce` runs the unit tests, recomputes every table above, regenerates the figure, and
writes `results/main.json`, `results/sweeps.json` and `results/criteria.md` (the pass/miss table is
computed, never hand-edited). Also: `python -m scripts.evaluate [--oracle] [--reid resnet18]`,
`python -m scripts.track --method ours`, `python -m scripts.render_video`. `ffmpeg` is used for
frame decoding and video encoding.

## Layout

```
tracker/    kalman.py assignment.py (own Hungarian) cost.py gallery.py lifecycle.py tracker.py
baselines/  greedy_iou.py kalman_iou.py appearance_only.py
eval/       hota.py idf1.py clear.py gaps.py taxonomy.py oracle.py report.py mot_io.py common.py
detection/  frozen_detector.py reid.py types.py
scripts/    fetch_data fetch_dets evaluate sweep reproduce track calibrate make_figures render_video ...
tests/      assignment (500 vs scipy) kalman hota gaps taxonomy baselines tracker tracker_parts
```

Not in the original layout but needed: `eval/clear.py` (CLEAR matching shared by MOTA, gaps and taxonomy
so all switch counts agree by construction), `eval/mot_io.py`, `eval/common.py`. Forbidden-dependency
rule respected: no `motmetrics`, `TrackEval`, `filterpy`, `deep_sort_realtime`, `boxmot`; scipy appears
only in `tests/test_assignment.py`.
