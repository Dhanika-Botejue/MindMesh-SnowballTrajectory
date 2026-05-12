import cv2
import math
from ultralytics import YOLO

# Class indices baked into runs/detect/hand-throw2/HandV2/weights/best.pt
# (model nc=8). The mapping is shifted on the right side relative to the
# original Roboflow ordering — indices 3 and 7 are stray classes from an
# earlier training run carried forward in the weights, and the throw-related
# right-side classes live at 4, 5, 6 (not 3, 4, 5).
# Arm classes (1, 2, 5, 6) bound the WHOLE arm shoulder->hand. Hand classes
# (0, 4) bound just the hand. Hand classes drive trajectory positions; arm
# classes drive the windup->finish state machine.
CLASS_NAMES = {
    0: "left hand",
    1: "left arm - finish throw",
    2: "left arm - windup",
    3: "left right",            # legacy, ignored
    4: "right hand",
    5: "right arm - finish throw",
    6: "right arm - windup",
    7: "right throw",           # legacy, ignored
}
CLASS_LEFT_HAND = 0
CLASS_LEFT_END = 1
CLASS_LEFT_START = 2
CLASS_RIGHT_HAND = 4
CLASS_RIGHT_END = 5
CLASS_RIGHT_START = 6

# Mac inference settings (CPU; smaller imgsz to avoid OOM/crashes).
INFER_DEVICE = "cpu"
INFER_IMGSZ = 416

# --- 3D world-space trajectory physics for an Insta360 equirectangular video ---
ASSUMED_THROW_DISTANCE_M = 0.5
CAMERA_HEIGHT_M = 1.5
GRAVITY_M_PER_S2 = 9.81
RELEASE_VELOCITY_GAIN = 7.0
LAUNCH_ANGLE_BOOST_DEG = 25.0
# Equirectangular projection compresses pixel motion along the camera's
# optical axis (+z in world space, which reprojects to image center). The
# inferred velocity therefore always under-counts the forward component on
# straight throws and over-counts lateral motion, so the trajectory veers
# off to the side. CENTER_BIAS_FACTOR rotates the horizontal velocity
# vector toward +z by the given fraction (0 = off / current behavior,
# 1 = throw entirely along +z) while preserving horizontal speed.
CENTER_BIAS_FACTOR = 0.7
TRAJ_STEP_S = 1.0 / 60.0
TRAJ_MAX_S = 4.0

POLYLINE_THICKNESS = 5
LANDING_DOT_RADIUS = 18
WINDUP_RELEASE_DOT_RADIUS = 16
LANDING_HOLD_FRAMES = 60
MAX_THROW_FRAMES = 20
MIN_THROW_FRAMES = 2

# Debug override: pin the predicted landing (red) dot to the center of the
# frame regardless of trajectory output. Useful for visual sanity checks.
HARDCODE_LANDING_TO_CENTER = True

# When True, prefer plain-hand bboxes (classes 0/3) for the windup and
# release pixels, with a temporal search window and a quadrant prior for the
# finish-throw fallback. Toggle off to A/B against the pure arm-center baseline.
USE_HAND_BBOX_REFINEMENT = True
HAND_SEARCH_WINDOW_FRAMES = 3


def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def finish_hand_quadrant(arm_bbox, hand):
    """Sub-region of an arm finish-throw bbox where the hand is known to sit:
    left hand -> top-right, right hand -> top-left. The thrower in this
    footage faces the camera and throws forward, so at release the hand is
    above the shoulder (top half) and on the outer side of the arm bbox."""
    x1, y1, x2, y2 = arm_bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if hand == "left":
        return (cx, y1, x2, cy)
    return (x1, y1, cx, cy)


def quadrant_center(quad):
    return ((quad[0] + quad[2]) / 2.0, (quad[1] + quad[3]) / 2.0)


def resolve_hand_pixel(detections, target_frame, hand_class,
                       containment_box, window):
    """Find a plain-hand bbox (class 0 or 3) near `target_frame` whose center
    lies inside `containment_box`. Searches the target frame first, then
    expands outward to +/- `window` frames. Returns (cx, cy) or None."""
    n = len(detections)
    for offset in range(0, window + 1):
        for f in {target_frame - offset, target_frame + offset}:
            if 0 <= f < n:
                xyxy, cls_arr, _ = detections[f]
                if xyxy is None:
                    continue
                for box, cls in zip(xyxy, cls_arr):
                    if int(cls) != hand_class:
                        continue
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    if (containment_box[0] <= cx <= containment_box[2]
                            and containment_box[1] <= cy <= containment_box[3]):
                        return (cx, cy)
    return None


def pixel_to_unit_ray(px, py, w, h):
    phi = (px / w) * 2 * math.pi - math.pi
    theta = math.pi / 2 - (py / h) * math.pi
    return (math.cos(theta) * math.sin(phi),
            math.sin(theta),
            math.cos(theta) * math.cos(phi))


def world_to_pixel(x, y, z, w, h):
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return (w / 2.0, h / 2.0)
    nx, ny, nz = x / n, y / n, z / n
    phi = math.atan2(nx, nz)
    theta = math.asin(max(-1.0, min(1.0, ny)))
    px = (phi + math.pi) / (2 * math.pi) * w
    py = (math.pi / 2 - theta) / math.pi * h
    return (px, py)


def simulate_trajectory(start_pos, release_pos, dt, w, h):
    if dt <= 0:
        return None
    sx, sy = start_pos
    ex, ey = release_pos

    R = ASSUMED_THROW_DISTANCE_M
    sd = pixel_to_unit_ray(sx, sy, w, h)
    ed = pixel_to_unit_ray(ex, ey, w, h)
    p_start = (R * sd[0], R * sd[1], R * sd[2])
    p_release = (R * ed[0], R * ed[1], R * ed[2])

    g = GRAVITY_M_PER_S2
    vx = (p_release[0] - p_start[0]) / dt * RELEASE_VELOCITY_GAIN
    vy = (p_release[1] - p_start[1]) / dt * RELEASE_VELOCITY_GAIN
    vz = (p_release[2] - p_start[2]) / dt * RELEASE_VELOCITY_GAIN

    # Real throws launch with upward loft; arm-swing tangent motion alone
    # gives near-flat vy and the ball dives too quickly. Add upward velocity
    # proportional to horizontal speed.
    horizontal_v = math.sqrt(vx * vx + vz * vz)
    vy += horizontal_v * math.tan(math.radians(LAUNCH_ANGLE_BOOST_DEG))

    ground_y = -CAMERA_HEIGHT_M
    points = [(int(ex), int(ey))]
    prev_y = p_release[1]
    t = 0.0
    while t < TRAJ_MAX_S:
        t += TRAJ_STEP_S
        x_t = p_release[0] + vx * t
        y_t = p_release[1] + vy * t - 0.5 * g * t * t
        z_t = p_release[2] + vz * t

        if y_t <= ground_y:
            denom = prev_y - y_t
            frac = (prev_y - ground_y) / denom if abs(denom) > 1e-9 else 1.0
            t_hit = (t - TRAJ_STEP_S) + frac * TRAJ_STEP_S
            x_hit = p_release[0] + vx * t_hit
            z_hit = p_release[2] + vz * t_hit
            px, py = world_to_pixel(x_hit, ground_y, z_hit, w, h)
            points.append((int(px), int(py)))
            break

        px, py = world_to_pixel(x_t, y_t, z_t, w, h)
        points.append((int(px), int(py)))
        prev_y = y_t

    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    horizontal = math.sqrt(vx * vx + vz * vz)
    angle_deg = math.degrees(math.atan2(vy, horizontal)) if horizontal > 1e-9 else 90.0
    return points, speed, angle_deg


class ThrowState:
    def __init__(self):
        self.start_bbox = None
        self.start_frame = None


def resolve_throw_pixels(detections, raw, frame_w, frame_h, fps):
    """Second-pass resolution of windup and release pixels for one raw throw
    event, then trajectory integration. Runs after the inference loop so the
    hand-bbox search window can look forward into post-release frames where
    the plain-hand class fires more reliably."""
    hand = raw["hand"]
    hand_id = CLASS_LEFT_HAND if hand == "left" else CLASS_RIGHT_HAND

    if USE_HAND_BBOX_REFINEMENT:
        windup_hit = resolve_hand_pixel(
            detections, raw["start_frame"], hand_id,
            raw["start_bbox"], HAND_SEARCH_WINDOW_FRAMES,
        )
        windup_xy = windup_hit or box_center(raw["start_bbox"])
        windup_src = "hand-bbox" if windup_hit else "arm-center"

        release_quad = finish_hand_quadrant(raw["end_bbox"], hand)
        release_hit = resolve_hand_pixel(
            detections, raw["end_frame"], hand_id,
            release_quad, HAND_SEARCH_WINDOW_FRAMES,
        )
        release_xy = release_hit or quadrant_center(release_quad)
        release_src = "hand-bbox" if release_hit else "quadrant"
    else:
        windup_xy = box_center(raw["start_bbox"])
        release_xy = box_center(raw["end_bbox"])
        windup_src = release_src = "arm-center"

    dt = (raw["end_frame"] - raw["start_frame"]) / fps
    sim = simulate_trajectory(windup_xy, release_xy, dt, frame_w, frame_h)
    if sim is None:
        return None
    polyline, speed, angle = sim
    print(
        f"  throw {hand}: dt={dt:.2f}s  "
        f"speed={speed:.2f}m/s  angle={angle:.0f}deg  "
        f"points={len(polyline)}  landing={polyline[-1]}  "
        f"windup={windup_src}  release={release_src}"
    )
    return {
        "start": raw["end_frame"],
        "end": raw["end_frame"] + LANDING_HOLD_FRAMES,
        "polyline": polyline,
        "landing": polyline[-1],
        "release": (int(release_xy[0]), int(release_xy[1])),
        "windup": (int(windup_xy[0]), int(windup_xy[1])),
        "hand": hand,
        "speed": speed,
        "angle": angle,
    }


def preprocess(model, source, fps, frame_w, frame_h):
    """Run YOLO inference once. Cache only per-frame detections (tiny) and
    throw events. Annotated frames are NOT cached — they're redrawn on demand
    in render() to avoid blowing out RAM on high-res 360 footage.

    Two-pass pipeline:
      1. Inference loop builds `detections` and a `raw_events` list of
         (start_frame, end_frame, hand, start_bbox, end_bbox) tuples driven
         by the windup -> finish-throw state machine on classes 1, 2, 4, 5.
      2. After the loop, each raw event resolves to precise windup/release
         pixels (preferring plain-hand bboxes 0/3 within HAND_SEARCH_WINDOW_FRAMES
         of the trigger frame, with the finish-throw quadrant prior as
         fallback) and integrates the trajectory. The two-pass split is what
         lets the release-side search look FORWARD into post-release frames,
         where motion blur subsides and the hand class fires reliably.
    """
    detections = []
    raw_events = []
    states = {"left": ThrowState(), "right": ThrowState()}

    results = model.predict(
        source=source,
        stream=True,
        conf=0.4,
        device=INFER_DEVICE,
        imgsz=INFER_IMGSZ,
        verbose=False,
    )

    for frame_idx, r in enumerate(results):
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            conf_arr = r.boxes.conf.cpu().numpy()
        else:
            xyxy, cls_arr, conf_arr = None, None, None
        detections.append((xyxy, cls_arr, conf_arr))

        best = {}
        if xyxy is not None:
            for box, cls, conf in zip(xyxy, cls_arr, conf_arr):
                if cls not in best or conf > best[cls][1]:
                    best[cls] = (box, conf)

        for hand, start_id, end_id in (
            ("left", CLASS_LEFT_START, CLASS_LEFT_END),
            ("right", CLASS_RIGHT_START, CLASS_RIGHT_END),
        ):
            st = states[hand]

            if start_id in best:
                st.start_bbox = best[start_id][0]
                st.start_frame = frame_idx

            if end_id in best and st.start_bbox is not None:
                if frame_idx - st.start_frame >= MIN_THROW_FRAMES:
                    raw_events.append({
                        "hand": hand,
                        "start_frame": st.start_frame,
                        "end_frame": frame_idx,
                        "start_bbox": st.start_bbox,
                        "end_bbox": best[end_id][0],
                    })
                st.start_bbox = None
                st.start_frame = None

            if (st.start_frame is not None
                    and frame_idx - st.start_frame > MAX_THROW_FRAMES):
                st.start_bbox = None
                st.start_frame = None

        if (frame_idx + 1) % 10 == 0:
            print(f"  processed {frame_idx + 1} frames")

    events = []
    for raw in raw_events:
        ev = resolve_throw_pixels(detections, raw, frame_w, frame_h, fps)
        if ev is not None:
            events.append(ev)

    return detections, events


def draw_polyline_safe(img, points, color, thickness, w):
    for i in range(len(points) - 1):
        if abs(points[i][0] - points[i + 1][0]) < w / 2:
            cv2.line(img, points[i], points[i + 1], color, thickness)


def parabolic_path(start, end, num_points=40):
    """Sample a downward-facing parabola from `start` to `end` in image
    coordinates (y grows downward). Apex sits above the straight-line
    interpolant with height proportional to the start->end distance."""
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    dist = math.sqrt(dx * dx + dy * dy)
    peak = max(40.0, dist * 0.35)
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        x = sx + t * dx
        y = sy + t * dy - peak * 4.0 * t * (1.0 - t)
        pts.append((int(x), int(y)))
    return pts


def clamp_point(p, w, h, margin):
    return (max(margin, min(w - margin - 1, int(p[0]))),
            max(margin, min(h - margin - 1, int(p[1]))))


def draw_detections(img, detection):
    """Manually redraw YOLO boxes from cached detection data."""
    xyxy, cls_arr, conf_arr = detection
    if xyxy is None:
        return
    for box, cls, conf in zip(xyxy, cls_arr, conf_arr):
        x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{CLASS_NAMES.get(int(cls), str(int(cls)))} {conf:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def render(frame, idx, total, events, detection):
    out = frame.copy()
    h, w = out.shape[:2]
    draw_detections(out, detection)
    for ev in events:
        if ev["start"] <= idx <= ev["end"]:
            windup = clamp_point(ev["windup"], w, h,
                                 WINDUP_RELEASE_DOT_RADIUS + 4)
            release = clamp_point(ev["release"], w, h,
                                  WINDUP_RELEASE_DOT_RADIUS + 4)
            if HARDCODE_LANDING_TO_CENTER:
                landing = (w // 2, h // 2 + h // 30)
                off_frame = False
                polyline = parabolic_path(release, landing)
            else:
                landing = clamp_point(ev["landing"], w, h, LANDING_DOT_RADIUS + 4)
                off_frame = landing != ev["landing"]
                polyline = ev["polyline"]
            draw_polyline_safe(
                out, polyline, (0, 165, 255), POLYLINE_THICKNESS, w
            )

            cv2.circle(out, windup, WINDUP_RELEASE_DOT_RADIUS + 3,
                       (255, 255, 255), 3)
            cv2.circle(out, windup, WINDUP_RELEASE_DOT_RADIUS,
                       (0, 255, 0), -1)
            cv2.circle(out, release, WINDUP_RELEASE_DOT_RADIUS + 3,
                       (255, 255, 255), 3)
            cv2.circle(out, release, WINDUP_RELEASE_DOT_RADIUS,
                       (255, 0, 0), -1)
            cv2.circle(out, landing, LANDING_DOT_RADIUS + 3,
                       (255, 255, 255), 2)
            cv2.circle(out, landing, LANDING_DOT_RADIUS, (0, 0, 255), -1)
            tag = "LAND (off-frame)" if off_frame else f"{ev['hand']} LAND"
            label = (f"{tag}  v={ev['speed']:.1f}m/s  "
                     f"a={int(ev['angle'])}deg")
            cv2.putText(
                out, label,
                (landing[0] + 20, landing[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
    cv2.putText(
        out, f"frame {idx + 1}/{total}   [2] next   [1] prev   [q] quit",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    return out


class FrameReader:
    """On-demand frame reader. Seeks via CAP_PROP_POS_FRAMES; fast on mp4
    with proper indexing and avoids holding every frame in RAM."""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.last_idx = -1

    def get(self, idx):
        if idx != self.last_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last_idx = idx
        return frame

    def release(self):
        self.cap.release()


def main():
    model = YOLO("runs/detect/hand-throw2/HandV2/weights/best.pt")
    # Override the names embedded in the .pt from training time (which still
    # carry the old "hand - start/end throw" strings). Patch only the known
    # indices instead of replacing the whole dict — the weights carry a
    # stale 7th class entry from earlier training and r.plot() crashes if
    # it emits an index whose name we dropped.
    for _idx, _name in CLASS_NAMES.items():
        model.model.names[_idx] = _name
    source = "snowball-3second-360.mp4"

    probe = cv2.VideoCapture(source)
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    print("Preprocessing video (running inference on every frame)...")
    detections, events = preprocess(model, source, fps, frame_w, frame_h)
    total = len(detections) if detections else total
    print(f"Done. {total} frames, {len(events)} throw event(s) detected.")
    print("Controls: 2 = next, 1 = previous, q = quit.")

    window_name = "Snowball Trajectory - Frame Stepper"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    reader = FrameReader(source)
    idx = 0
    while True:
        frame = reader.get(idx)
        if frame is None:
            break
        cv2.imshow(window_name, render(frame, idx, total, events, detections[idx]))
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('2'):
            idx = min(idx + 1, total - 1)
        elif key == ord('1'):
            idx = max(idx - 1, 0)

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
