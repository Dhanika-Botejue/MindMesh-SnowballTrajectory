# MindMesh-SnowballTrajectory

**Predict where a thrown snowball will land from Insta360 video, using hand and arm throw trajectory calculation.**

![demo](model-demo.gif)

## Approach

A YOLOv8 detector trained on 500+ annotated frames recognizes six classes per frame: `left/right hand` (just the hand), `left/right arm - windup` (the whole arm at the start of a throw, shoulder→hand), and `left/right arm - finish throw` (the whole arm at release). The trajectory pipeline runs on top of that detector:

1. **Per-hand state machine** — for each hand (left, right), the first `arm - windup` detection locks in the windup arm bbox and frame index; the next `arm - finish throw` detection is treated as the release. A windup that never resolves into a finish-throw is dropped after a timeout. Arm classes drive *timing* only — they decide when the throw starts and ends.
2. **Two-signal pixel resolution** — the windup and release **pixels** are sourced from the plain-hand classes (0, 3), not from the arm bbox centers. For each event:
   - **Preferred**: take the matching-side hand bbox center on the trigger frame, accept it only if it lies within the corresponding arm bbox (windup) or the expected release quadrant (finish throw). If no hand bbox fires on the trigger frame, expand outward to ±`HAND_SEARCH_WINDOW_FRAMES` frames; the closest in time wins. The two-pass design (state machine first, pixel resolution second) lets the release-side search look *forward* into post-release frames, where motion blur subsides and the plain-hand class fires more reliably.
   - **Finish-throw fallback** uses a tight spatial prior derived from this footage: the **left hand at release always sits in the top-right quadrant** of the left-arm finish-throw bbox, and the **right hand at release always sits in the top-left quadrant** of the right-arm finish-throw bbox (the thrower faces the camera and throws forward, so the hand is above the shoulder and on the outer side of the arm at release). When no plain-hand bbox is found in the search window, the release pixel falls back to the centroid of that quadrant — strictly better than arm-bbox center, because it uses a trained-data-derived spatial prior rather than a generic geometric heuristic.
   - **Windup fallback** is the arm bbox center; windup pose is more variable and no analogous quadrant prior is asserted.
   - The toggle `USE_HAND_BBOX_REFINEMENT` reverts to the old arm-center-to-arm-center behavior for A/B comparison.
3. **Equirectangular unprojection to 3D** — Insta360 footage is exported as an equirectangular projection (`x → azimuth φ ∈ [-π, π]`, `y → elevation θ ∈ [π/2, -π/2]`). The resolved windup and release pixels are each converted to a unit ray and lifted onto a sphere of radius `ASSUMED_THROW_DISTANCE_M` around the camera, giving a 3D world point for both.
4. **Initial 3D velocity** — `v = (release₃D − windup₃D) / Δt`, multiplied by `RELEASE_VELOCITY_GAIN` to scale the average windup-to-release motion up to a peak release speed. `Δt` comes from the source video's FPS. This avoids the "constant angular velocity" failure mode where forward throws appear to sweep across the entire frame, because angular velocity now correctly *decreases* as the ball moves further from the camera.
5. **3D projectile model** — the throw is integrated in world space with real gravity acting on `+y` (down):
   - `p(t) = p_release + v · t + ½ · g_vec · t²` where `g_vec = (0, −9.81, 0)`
   At each timestep the 3D position is reprojected back to an equirectangular pixel via `world_to_pixel`. The simulation stops when the world-`y` coordinate crosses `−CAMERA_HEIGHT_M` (the assumed ground), with linear interpolation to the exact crossing point.
6. **Visualization** — interactive frame stepper. Orange polyline traces the curved trajectory from release to landing; a red filled dot (white-ringed) marks the predicted impact point; green and blue dots mark the resolved windup and release pixels; an overlay shows release speed (m/s) and launch angle. The per-throw console line additionally prints the *source* of each pixel (`hand-bbox` / `arm-center` / `quadrant`) so you can tell at a glance when a throw fell back to a fallback path. Polyline segments that cross the equirectangular seam are skipped to avoid spurious cross-frame lines, and any marker whose true coordinate falls outside the frame is clamped to the nearest edge so it always renders.

### Controls

The viewer pre-runs inference over the whole clip, then drops into a manual stepper:

- `2` — next frame
- `1` — previous frame
- `q` — quit

### Tunable constants (`test_model.py`)

- `ASSUMED_THROW_DISTANCE_M` — meters from the camera to the thrower's hand. Default `2.0`. Sets the radius of the sphere we lift the hand pixels onto. Larger → larger absolute throw velocities for the same angular hand motion → longer trajectories.
- `CAMERA_HEIGHT_M` — meters from the camera to the ground. Default `1.5`. The trajectory terminates when world `y` drops below `−CAMERA_HEIGHT_M`.
- `GRAVITY_M_PER_S2` — real gravity, default `9.81`. No reason to change unless modelling another planet.
- `RELEASE_VELOCITY_GAIN` — multiplier on the windup→release average velocity to approximate peak release speed. Default `3.0`. Raise if throws fall too short, lower if they overshoot.
- `USE_HAND_BBOX_REFINEMENT` — when `True` (default), windup/release pixels come from the plain-hand classes with the temporal search window and finish-throw quadrant prior described above; when `False`, they revert to the raw arm bbox centers (legacy behavior). A/B-toggle this on the same clip to measure how much the refinement changes predicted speed and landing.
- `HAND_SEARCH_WINDOW_FRAMES` — how many frames forward and backward of the windup/finish trigger to search for a co-firing plain-hand detection. Default `3` (≈50 ms at 60 fps — long enough to bridge a single missed detection without picking up an unrelated hand pose).
- `MIN_THROW_FRAMES` / `MAX_THROW_FRAMES` — minimum frame gap for a valid `start→end` pairing, and how long a stale windup is held before being cleared.
- `POLYLINE_THICKNESS` / `LANDING_DOT_RADIUS` — visibility knobs for the orange path and the red landing dot.
- `LANDING_HOLD_FRAMES` — how many frames the landing dot stays visible after release.

## Results

- Trained YOLOv8n on 500+ annotated frames, evaluated on a held-out validation split — the detector reliably localizes both hands and fires `arm - windup` / `arm - finish throw` events at the correct moments in real-world outdoor footage.
- The full detect → angular-velocity → equirectangular-trajectory pipeline runs frame-by-frame on RTX 3060 Ti hardware at real-time playback rates; the macOS frame stepper preprocesses the clip on CPU and then advances at user-controlled pace.
- Predicted landing dot tracks observed snowball impact points across multiple test throws after one-time calibration of the angular gravity and ground-elevation constants.

## Context

Built as part of BCI-XR research under Prof. Steve Mann at the University of Toronto (Feb 2026 – present), contributing to an in-progress IEEE CBMS 2026 paper, *"Wearable BCI-XR for Adjunctive Therapy in Drug Addiction Rehab.: Collective Neurophysiological Synchrony During Cold Exposure"*. Snowball throws are the in-game action of the paper's *Buzzkill* XR module, where participants' cold-induced physiological responses launch virtual snowballs to extinguish flames as part of a closed-loop, group-cold-exposure biofeedback system.
