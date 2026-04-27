import cv2
import math
from ultralytics import YOLO

# Class indices from hand-throw-dataset/data.yaml:
# 0 left hand, 1 left end, 2 left start, 3 right hand, 4 right end, 5 right start
CLASS_LEFT_END = 1
CLASS_LEFT_START = 2
CLASS_RIGHT_END = 4
CLASS_RIGHT_START = 5

# --- Equirectangular (Insta360) trajectory physics ---
# Insta360 video exports as equirectangular: horizontal pixels map to azimuth
# phi in [-pi, pi], vertical pixels map to elevation theta in [pi/2, -pi/2].
# A real-world straight-line throw becomes a curve in this projection, so we
# simulate in angular space and reproject the curve back to pixels.
GRAVITY_RAD_PER_S2 = 1.0          # angular gravity; tune until arc matches reality
GROUND_PIXEL_Y_FRAC = 0.92        # treat this fraction down the frame as "ground"
RELEASE_VELOCITY_GAIN = 1.2       # peak release speed ~ this * windup-to-release average
LAUNCH_ELEVATION_BOOST = 0.2      # rad/s upward kick added to v_theta at release
TRAJ_STEP_S = 1.0 / 60.0          # simulation step size
TRAJ_MAX_S = 4.0                  # cap simulation length

POLYLINE_THICKNESS = 5
LANDING_DOT_RADIUS = 18
LANDING_HOLD_FRAMES = 60
MAX_THROW_FRAMES = 20         # clear a stale windup that never resolves
MIN_THROW_FRAMES = 2          # require this many frames between start and end


def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def pixel_to_angles(px, py, w, h):
    phi = (px / w) * 2 * math.pi - math.pi
    theta = math.pi / 2 - (py / h) * math.pi
    return phi, theta


def angles_to_pixel(phi, theta, w, h):
    px = (phi + math.pi) / (2 * math.pi) * w
    py = (math.pi / 2 - theta) / math.pi * h
    return px, py


def simulate_trajectory(start_pos, release_pos, dt, w, h):
    """Simulate the throw in angular space and return (polyline, speed, angle).
    Polyline is a list of pixel points from release to predicted landing."""
    if dt <= 0:
        return None
    sx, sy = start_pos
    ex, ey = release_pos
    s_phi, s_theta = pixel_to_angles(sx, sy, w, h)
    e_phi, e_theta = pixel_to_angles(ex, ey, w, h)

    v_phi = (e_phi - s_phi) / dt        # rad/s azimuth
    v_theta = (e_theta - s_theta) / dt  # rad/s elevation

    # Peak release velocity is well above the average windup-to-release velocity,
    # and a typical throw launches slightly upward even when the arm motion is
    # roughly horizontal. Apply a multiplicative gain plus an additive elevation
    # boost so the simulation actually sails forward instead of immediately
    # diving into the ground.
    v_phi *= RELEASE_VELOCITY_GAIN
    v_theta = v_theta * RELEASE_VELOCITY_GAIN + LAUNCH_ELEVATION_BOOST

    _, ground_theta = pixel_to_angles(0, GROUND_PIXEL_Y_FRAC * h, w, h)

    points = [(int(ex), int(ey))]
    prev_theta = e_theta
    t = 0.0
    while t < TRAJ_MAX_S:
        t += TRAJ_STEP_S
        theta_t = e_theta + v_theta * t - 0.5 * GRAVITY_RAD_PER_S2 * t * t
        phi_t = e_phi + v_phi * t

        if theta_t <= ground_theta:
            denom = prev_theta - theta_t
            frac = (prev_theta - ground_theta) / denom if abs(denom) > 1e-9 else 1.0
            t_hit = (t - TRAJ_STEP_S) + frac * TRAJ_STEP_S
            phi_hit = e_phi + v_phi * t_hit
            px, py = angles_to_pixel(phi_hit, ground_theta, w, h)
            points.append((int(px), int(py)))
            break

        px, py = angles_to_pixel(phi_t, theta_t, w, h)
        points.append((int(px), int(py)))
        prev_theta = theta_t

    speed_rad = math.hypot(v_phi, v_theta)
    angle_deg = math.degrees(math.atan2(-v_theta, v_phi))
    return points, speed_rad, angle_deg


class ThrowState:
    def __init__(self):
        self.start_pos = None
        self.start_frame = None


def preprocess(model, source, fps):
    cached = []
    events = []
    states = {"left": ThrowState(), "right": ThrowState()}

    results = model.predict(
        source=source,
        stream=True,
        conf=0.4,
        device="cpu",
        imgsz=416,
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

            # Always overwrite with the most recent start_throw. This avoids
            # locking onto a stale early false-positive and uses the windup
            # position closest in time to the actual release.
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
                            f"speed={speed:.2f}rad/s  angle={angle:.0f}deg  "
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
    """Draw a polyline but skip segments that wrap across the equirectangular
    seam (i.e. where x jumps by more than half the frame width)."""
    for i in range(len(points) - 1):
        if abs(points[i][0] - points[i + 1][0]) < w / 2:
            cv2.line(img, points[i], points[i + 1], color, thickness)


def clamp_point(p, w, h, margin):
    """OpenCV silently skips circles whose center is outside the image.
    Clamp markers to within the frame so they always render."""
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

            # Windup (green) and release (blue) input markers with white outlines.
            cv2.circle(out, windup, 10, (255, 255, 255), 2)
            cv2.circle(out, windup, 8, (0, 255, 0), -1)
            cv2.circle(out, release, 10, (255, 255, 255), 2)
            cv2.circle(out, release, 8, (255, 0, 0), -1)
            # White ring + filled red center keeps the landing dot visible
            # against any color underneath (polyline, YOLO box, snow).
            cv2.circle(out, landing, LANDING_DOT_RADIUS + 3,
                       (255, 255, 255), 2)
            cv2.circle(out, landing, LANDING_DOT_RADIUS, (0, 0, 255), -1)
            tag = "LAND (off-frame)" if off_frame else f"{ev['hand']} LAND"
            label = (f"{tag}  w={ev['speed']:.2f}rad/s  "
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
