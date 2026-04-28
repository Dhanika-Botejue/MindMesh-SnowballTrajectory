import cv2
import math
from ultralytics import YOLO

# Class indices from hand-throw-dataset/data.yaml:
# 0 left hand, 1 left end, 2 left start, 3 right hand, 4 right end, 5 right start
CLASS_NAMES = {
    0: "left hand",
    1: "left end",
    2: "left start",
    3: "right hand",
    4: "right end",
    5: "right start",
}
CLASS_LEFT_END = 1
CLASS_LEFT_START = 2
CLASS_RIGHT_END = 4
CLASS_RIGHT_START = 5

# Windows / RTX inference settings.
INFER_DEVICE = 0
INFER_IMGSZ = 640

# --- 3D world-space trajectory physics for an Insta360 equirectangular video ---
ASSUMED_THROW_DISTANCE_M = 0.5
CAMERA_HEIGHT_M = 1.5
GRAVITY_M_PER_S2 = 9.81
RELEASE_VELOCITY_GAIN = 7.0
LAUNCH_ANGLE_BOOST_DEG = 25.0
TRAJ_STEP_S = 1.0 / 60.0
TRAJ_MAX_S = 4.0

POLYLINE_THICKNESS = 5
LANDING_DOT_RADIUS = 18
LANDING_HOLD_FRAMES = 60
MAX_THROW_FRAMES = 20
MIN_THROW_FRAMES = 2


def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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
        self.start_pos = None
        self.start_frame = None


def preprocess(model, source, fps):
    """Cache `r.plot()` output per frame. On Windows / RTX the GPU pipeline
    and typically more system RAM make this practical; on Mac use the
    on-demand reader in test_model.py instead."""
    cached = []
    events = []
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
        annotated = r.plot()
        h, w = annotated.shape[:2]

        best = {}
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            conf_arr = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(xyxy, cls_arr, conf_arr):
                if cls not in best or conf > best[cls][1]:
                    best[cls] = (box, conf)

        for hand, start_id, end_id in (
            ("left", CLASS_LEFT_START, CLASS_LEFT_END),
            ("right", CLASS_RIGHT_START, CLASS_RIGHT_END),
        ):
            st = states[hand]

            if start_id in best:
                st.start_pos = box_center(best[start_id][0])
                st.start_frame = frame_idx

            if end_id in best and st.start_pos is not None:
                if frame_idx - st.start_frame >= MIN_THROW_FRAMES:
                    release = box_center(best[end_id][0])
                    dt = (frame_idx - st.start_frame) / fps
                    windup = (int(st.start_pos[0]), int(st.start_pos[1]))
                    sim = simulate_trajectory(st.start_pos, release, dt, w, h)
                    if sim is not None:
                        polyline, speed, angle = sim
                        print(
                            f"  throw {hand}: dt={dt:.2f}s  "
                            f"speed={speed:.2f}m/s  angle={angle:.0f}deg  "
                            f"points={len(polyline)}  landing={polyline[-1]}"
                        )
                        events.append({
                            "start": frame_idx,
                            "end": frame_idx + LANDING_HOLD_FRAMES,
                            "polyline": polyline,
                            "landing": polyline[-1],
                            "release": (int(release[0]), int(release[1])),
                            "windup": windup,
                            "hand": hand,
                            "speed": speed,
                            "angle": angle,
                        })
                st.start_pos = None
                st.start_frame = None

            if (st.start_frame is not None
                    and frame_idx - st.start_frame > MAX_THROW_FRAMES):
                st.start_pos = None
                st.start_frame = None

        cached.append(annotated)
        if (frame_idx + 1) % 10 == 0:
            print(f"  processed {frame_idx + 1} frames")

    return cached, events


def draw_polyline_safe(img, points, color, thickness, w):
    for i in range(len(points) - 1):
        if abs(points[i][0] - points[i + 1][0]) < w / 2:
            cv2.line(img, points[i], points[i + 1], color, thickness)


def clamp_point(p, w, h, margin):
    return (max(margin, min(w - margin - 1, int(p[0]))),
            max(margin, min(h - margin - 1, int(p[1]))))


def render(frame, idx, total, events):
    out = frame.copy()
    h, w = out.shape[:2]
    for ev in events:
        if ev["start"] <= idx <= ev["end"]:
            draw_polyline_safe(
                out, ev["polyline"], (0, 165, 255), POLYLINE_THICKNESS, w
            )
            windup = clamp_point(ev["windup"], w, h, 8)
            release = clamp_point(ev["release"], w, h, 8)
            landing = clamp_point(ev["landing"], w, h, LANDING_DOT_RADIUS + 4)
            off_frame = landing != ev["landing"]

            cv2.circle(out, windup, 10, (255, 255, 255), 2)
            cv2.circle(out, windup, 8, (0, 255, 0), -1)
            cv2.circle(out, release, 10, (255, 255, 255), 2)
            cv2.circle(out, release, 8, (255, 0, 0), -1)
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


def main():
    model = YOLO("runs/detect/hand-throw2/HandV2/weights/best.pt")
    source = "snowball-3second-360.mp4"

    probe = cv2.VideoCapture(source)
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()

    print("Preprocessing video (running inference on every frame)...")
    cached, events = preprocess(model, source, fps)
    total = len(cached)
    print(f"Done. {total} frames cached, {len(events)} throw event(s) detected.")
    print("Controls: 2 = next, 1 = previous, q = quit.")

    window_name = "Snowball Trajectory - Frame Stepper"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    idx = 0
    while True:
        cv2.imshow(window_name, render(cached[idx], idx, total, events))
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('2'):
            idx = min(idx + 1, total - 1)
        elif key == ord('1'):
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
