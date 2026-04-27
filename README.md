# MindMesh-SnowballTrajectory

**Predict where a thrown snowball will land from Insta360 video, using only hand detections.**

<!-- TODO: drop a short GIF / frame sequence of the red landing dot tracking a real throw here -->
![demo placeholder](docs/demo.gif)

## Approach

A YOLOv8 detector trained on 500+ annotated frames recognizes six hand states per frame: `left/right hand`, `left/right hand - start throw`, `left/right hand - end throw`. The trajectory pipeline runs on top of that detector:

1. **Per-hand state machine** — for each hand (left, right), the first `start throw` detection locks in the windup position and frame index; the next `end throw` detection is treated as the release point. A windup that never resolves into an end-throw is dropped after a timeout.
2. **Equirectangular unprojection to 3D** — Insta360 footage is exported as an equirectangular projection (`x → azimuth φ ∈ [-π, π]`, `y → elevation θ ∈ [π/2, -π/2]`). Each hand pixel is converted to a unit ray and lifted onto a sphere of radius `ASSUMED_THROW_DISTANCE_M` around the camera, giving a 3D world point for both the windup and the release.
3. **Initial 3D velocity** — `v = (release₃D − windup₃D) / Δt`, multiplied by `RELEASE_VELOCITY_GAIN` to scale the average windup-to-release motion up to a peak release speed. `Δt` comes from the source video's FPS. This avoids the "constant angular velocity" failure mode where forward throws appear to sweep across the entire frame, because angular velocity now correctly *decreases* as the ball moves further from the camera.
4. **3D projectile model** — the throw is integrated in world space with real gravity acting on `+y` (down):
   - `p(t) = p_release + v · t + ½ · g_vec · t²` where `g_vec = (0, −9.81, 0)`
   At each timestep the 3D position is reprojected back to an equirectangular pixel via `world_to_pixel`. The simulation stops when the world-`y` coordinate crosses `−CAMERA_HEIGHT_M` (the assumed ground), with linear interpolation to the exact crossing point.
5. **Visualization** — interactive frame stepper. Orange polyline traces the curved trajectory from release to landing; a red filled dot (white-ringed) marks the predicted impact point; green and blue dots mark the model-detected windup and release positions; an overlay shows release speed (m/s) and launch angle. Polyline segments that cross the equirectangular seam are skipped to avoid spurious cross-frame lines, and any marker whose true coordinate falls outside the frame is clamped to the nearest edge so it always renders.

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
- `MIN_THROW_FRAMES` / `MAX_THROW_FRAMES` — minimum frame gap for a valid `start→end` pairing, and how long a stale windup is held before being cleared.
- `POLYLINE_THICKNESS` / `LANDING_DOT_RADIUS` — visibility knobs for the orange path and the red landing dot.
- `LANDING_HOLD_FRAMES` — how many frames the landing dot stays visible after release.

## Results

- Trained YOLOv8n on 500+ annotated frames, evaluated on a held-out validation split — the detector reliably localizes both hands and fires `start throw` / `end throw` events at the correct moments in real-world outdoor footage.
- The full detect → angular-velocity → equirectangular-trajectory pipeline runs frame-by-frame on RTX 3060 Ti hardware at real-time playback rates; the macOS frame stepper preprocesses the clip on CPU and then advances at user-controlled pace.
- Predicted landing dot tracks observed snowball impact points across multiple test throws after one-time calibration of the angular gravity and ground-elevation constants.

## Context

Built as part of BCI-XR research under Prof. Steve Mann at the University of Toronto (Feb 2026 – present), contributing to an in-progress IEEE CBMS 2026 paper, *"Wearable BCI-XR for Adjunctive Therapy in Drug Addiction Rehab.: Collective Neurophysiological Synchrony During Cold Exposure"*. Snowball throws are the in-game action of the paper's *Buzzkill* XR module, where participants' cold-induced physiological responses launch virtual snowballs to extinguish flames as part of a closed-loop, group-cold-exposure biofeedback system.
