# MindMesh-SnowballTrajectory

**Predict where a thrown snowball will land from monocular video, using only hand detections.**

<!-- TODO: drop a short GIF / frame sequence of the red landing dot tracking a real throw here -->
![demo placeholder](docs/demo.gif)

## Approach

A YOLOv8 detector trained on 500+ annotated frames recognizes six hand states per frame: `left/right hand`, `left/right hand - start throw`, `left/right hand - end throw`. The trajectory pipeline runs on top of that detector:

1. **Per-hand state machine** — for each hand (left, right), the first `start throw` detection locks in the windup position and frame index; the next `end throw` detection is treated as the release point. A windup that never resolves into an end-throw is dropped after a timeout.
2. **Initial velocity** — `(release − start) / Δt` in pixels per second, where `Δt` comes from the source video's FPS. This gives both horizontal (`vx`) and vertical (`vy`) components in image space.
3. **Launch angle** — `atan2(-vy, vx)`, with `vy` negated because image-space y points down while world-space up is positive.
4. **Projectile model** — image-space kinematics with gravity acting along +y:
   - `x(t) = x_release + vx·t`
   - `y(t) = y_release + vy·t + ½·g·t²`
   The landing time is the positive root of `½·g·t² + vy·t + (y_release − y_ground) = 0`. Ground is the bottom of the frame; gravity is a tunable pixel-space constant calibrated to camera distance and resolution.
5. **Visualization** — an orange arrow from the release point to the predicted landing, a red filled dot at the landing, and an overlay of speed (px/s) and launch angle.

## Results

- Trained YOLOv8n on 500+ annotated frames, evaluated on a held-out validation split — the detector reliably localizes both hands and fires `start throw` / `end throw` events at the correct moments in real-world outdoor footage.
- The full detect → velocity → trajectory pipeline runs frame-by-frame on RTX 3060 Ti hardware at real-time playback rates.
- Predicted landing dot tracks observed snowball impact points across multiple test throws after one-time calibration of the pixel-space gravity constant.

## Context

Built as part of BCI-XR research under Prof. Steve Mann at the University of Toronto, contributing to an in-progress IEEE CBMS 2026 paper on *Wearable BCI-XR Technology For Adjunctive Therapy In Drug Addiction Rehab*.
