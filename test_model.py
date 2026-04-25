import cv2
import math
from ultralytics import YOLO

# Class indices from hand-throw-dataset/data.yaml:
# 0 left hand, 1 left end, 2 left start, 3 right hand, 4 right end, 5 right start
CLASS_LEFT_END = 1
CLASS_LEFT_START = 2
CLASS_RIGHT_END = 4
CLASS_RIGHT_START = 5

# Pixel-space gravity. Tune until predicted landing matches reality
# for your camera distance / resolution. ~9.81 m/s^2 * pixels_per_meter.
GRAVITY_PX_PER_S2 = 1500.0
LANDING_HOLD_FRAMES = 60     # how long the red dot stays on screen
MAX_THROW_FRAMES = 60        # abandon a windup that never resolves


def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def predict_landing(start_pos, end_pos, dt, ground_y, g):
    """Forward-simulate projectile motion in image-space coordinates
    (y increases downward) and return the (x, y) where it crosses ground_y."""
    if dt <= 0:
        return None
    sx, sy = start_pos
    ex, ey = end_pos
    vx = (ex - sx) / dt
    vy = (ey - sy) / dt

    # y(t) = ey + vy*t + 0.5*g*t^2 = ground_y
    a = 0.5 * g
    b = vy
    c = ey - ground_y
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t_land = (-b + math.sqrt(disc)) / (2 * a)
    if t_land <= 0:
        return None
    landing_x = ex + vx * t_land
    speed = math.hypot(vx, vy)
    angle_deg = math.degrees(math.atan2(-vy, vx))  # negate vy: world up is positive
    return (int(landing_x), int(ground_y)), speed, angle_deg


class ThrowState:
    def __init__(self):
        self.start_pos = None
        self.start_frame = None
        self.release_pos = None
        self.landing = None
        self.hold = 0
        self.speed = 0.0
        self.angle = 0.0


def main():
    model = YOLO(r"runs\detect\hand-throw2\HandV2\weights\best.pt")
    source = "snowball-3second-360.mp4"

    probe = cv2.VideoCapture(source)
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()

    window_name = "Snowball Trajectory - Landing Prediction"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    states = {"left": ThrowState(), "right": ThrowState()}
    frame_idx = 0

    results = model.predict(source=source, stream=True, device=0, conf=0.4)

    for r in results:
        annotated = r.plot()
        h = annotated.shape[0]
        ground_y = h - 1

        # Pick the highest-confidence box per class on this frame.
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

            # Lock in the first start-throw frame of a new throw.
            if start_id in best and st.start_pos is None:
                st.start_pos = box_center(best[start_id][0])
                st.start_frame = frame_idx

            # On end-throw, treat that hand position as release point and predict.
            if end_id in best and st.start_pos is not None:
                release = box_center(best[end_id][0])
                dt = (frame_idx - st.start_frame) / fps
                result = predict_landing(
                    st.start_pos, release, dt, ground_y, GRAVITY_PX_PER_S2
                )
                if result is not None:
                    st.landing, st.speed, st.angle = result
                    st.release_pos = release
                    st.hold = LANDING_HOLD_FRAMES
                st.start_pos = None
                st.start_frame = None

            # Drop a stuck windup that never produced an end-throw.
            if (st.start_frame is not None
                    and frame_idx - st.start_frame > MAX_THROW_FRAMES):
                st.start_pos = None
                st.start_frame = None

            if st.hold > 0 and st.landing is not None:
                if st.release_pos is not None:
                    rp = (int(st.release_pos[0]), int(st.release_pos[1]))
                    cv2.arrowedLine(annotated, rp, st.landing, (0, 165, 255), 2)
                cv2.circle(annotated, st.landing, 12, (0, 0, 255), -1)
                label = f"{hand}  v={int(st.speed)}px/s  a={int(st.angle)}deg"
                cv2.putText(
                    annotated, label,
                    (st.landing[0] + 15, st.landing[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                )
                st.hold -= 1

        cv2.imshow(window_name, annotated)
        frame_idx += 1
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
