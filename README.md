# MindMesh-SnowballTrajectory

**Predict where a thrown snowball will land from Insta360 video, using only hand detections.**

<!-- TODO: drop a short GIF / frame sequence of the red landing dot tracking a real throw here -->
![demo placeholder](docs/demo.gif)

## Approach

A YOLOv8 detector trained on 500+ annotated frames recognizes six hand states per frame: `left/right hand`, `left/right hand - start throw`, `left/right hand - end throw`. The trajectory pipeline runs on top of that detector:

1. **Per-hand state machine** — for each hand (left, right), the first `start throw` detection locks in the windup position and frame index; the next `end throw` detection is treated as the release point. A windup that never resolves into an end-throw is dropped after a timeout.
2. **Equirectangular unprojection** — Insta360 footage is exported as an equirectangular projection: horizontal pixels map to azimuth `φ ∈ [-π, π]`, vertical pixels map to elevation `θ ∈ [π/2, -π/2]`. A real-world straight-line throw becomes a curve in pixel space, so the windup and release positions are first converted from pixels to angular coordinates `(φ, θ)`.
3. **Initial angular velocity** — `(φ_release − φ_start) / Δt` and `(θ_release − θ_start) / Δt` give azimuthal and elevational velocities in rad/s, with `Δt` derived from the source video's FPS.
4. **Angular-space projectile model** — the throw is forward-simulated on the sphere with gravity acting on elevation:
   - `φ(t) = φ_release + ω_φ · t`
   - `θ(t) = θ_release + ω_θ · t − ½ · g · t²`
   The simulation steps in small time increments until elevation crosses a configurable ground angle, then linearly interpolates the exact crossing point. Each `(φ, θ)` is reprojected back to pixel space, producing a polyline that follows the equirectangular curvature instead of a fictitious pixel-space parabola.
5. **Visualization** — interactive frame stepper. Orange polyline traces the curved trajectory from release to landing; a red filled dot marks the predicted impact point; an overlay shows angular speed (rad/s) and launch angle. Polyline segments that cross the equirectangular seam are skipped to avoid spurious cross-frame lines.

### Controls

The viewer pre-runs inference over the whole clip, then drops into a manual stepper:

- `2` — next frame
- `1` — previous frame
- `q` — quit

### Tunable constants (`test_model.py`)

- `GRAVITY_RAD_PER_S2` — angular gravity in rad/s². Depends on assumed throw distance from camera (`g_real / D`). Default `1.0`. Lower → longer flight before hitting ground.
- `GROUND_PIXEL_Y_FRAC` — fraction of frame height treated as ground. Default `0.92`. Raise to push the assumed ground further down the image.
- `RELEASE_VELOCITY_GAIN` — multiplier on the windup→release average velocity to approximate the peak release speed. Default `1.8`.
- `LAUNCH_ELEVATION_BOOST` — additive upward kick (rad/s) on `v_θ` at release so the arc lifts even when the arm motion is roughly horizontal. Default `0.5`.
- `POLYLINE_THICKNESS` / `LANDING_DOT_RADIUS` — visibility knobs for the orange path and the red landing dot.
- `LANDING_HOLD_FRAMES` — how many frames the landing dot stays visible after release.

## Results

- Trained YOLOv8n on 500+ annotated frames, evaluated on a held-out validation split — the detector reliably localizes both hands and fires `start throw` / `end throw` events at the correct moments in real-world outdoor footage.
- The full detect → angular-velocity → equirectangular-trajectory pipeline runs frame-by-frame on RTX 3060 Ti hardware at real-time playback rates; the macOS frame stepper preprocesses the clip on CPU and then advances at user-controlled pace.
- Predicted landing dot tracks observed snowball impact points across multiple test throws after one-time calibration of the angular gravity and ground-elevation constants.

## Context

Built as part of BCI-XR research under Prof. Steve Mann at the University of Toronto (Feb 2026 – present), contributing to an in-progress IEEE CBMS 2026 paper, *"Wearable BCI-XR for Adjunctive Therapy in Drug Addiction Rehab.: Collective Neurophysiological Synchrony During Cold Exposure"*. Snowball throws are the in-game action of the paper's *Buzzkill* XR module, where participants' cold-induced physiological responses launch virtual snowballs to extinguish flames as part of a closed-loop, group-cold-exposure biofeedback system.
