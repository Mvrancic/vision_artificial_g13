import argparse
import os
import platform
import shutil
import subprocess
import sys
import threading
import wave

cv2 = None
np = None


def ensure_runtime_python():
    global cv2, np
    try:
        import cv2 as _cv2  # type: ignore
        import numpy as _np  # type: ignore
        cv2 = _cv2
        np = _np
        return
    except ImportError:
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    venv_python = os.path.join(project_root, ".venv", "bin", "python")

    if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
        os.execv(venv_python, [venv_python, __file__, *sys.argv[1:]])

    raise ImportError(
        "Missing dependencies. Install cv2 + mediapipe, or run with .venv/bin/python."
    )


def import_real_mediapipe():
    # Avoid importing the local project folder named "mediapipe".
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    blocked_paths = {script_dir, project_root, ""}
    original_sys_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p not in blocked_paths]
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mediapipe package is not installed. Run: python3 -m pip install mediapipe"
        ) from exc
    finally:
        sys.path = original_sys_path
    return mp


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_image = os.path.join(script_dir, "absolute-cinema.webp")
    default_freaky_image = os.path.join(script_dir, "absolute-cinema-absolute-freaky.png")
    default_cinema_sound = os.path.join(script_dir, "explosion.wav")
    default_freaky_sound = os.path.join(script_dir, "scream.wav")
    parser = argparse.ArgumentParser(
        description="Show a selected image when motion is detected and both hands are raised."
    )
    parser.add_argument(
        "--image",
        default=default_image,
        help="Path to the image to display when both hands are up.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0).",
    )
    parser.add_argument(
        "--freaky-image",
        default=default_freaky_image,
        help="Path to the image to display when both hands are up and tongue is out.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=int,
        default=4000,
        help="Minimum changed pixels to consider motion (default: 4000).",
    )
    parser.add_argument(
        "--cinema-sound",
        default=default_cinema_sound,
        help="Sound file path to play when cinema image appears.",
    )
    parser.add_argument(
        "--freaky-sound",
        default=default_freaky_sound,
        help="Sound file path to play when freaky image appears.",
    )
    return parser.parse_args()


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def detect_motion(prev_gray, gray, motion_threshold):
    frame_delta = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    changed_pixels = cv2.countNonZero(thresh)
    return changed_pixels > motion_threshold, changed_pixels


def hands_up(pose_landmarks, mp_pose):
    lm = pose_landmarks.landmark

    left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    visibility_ok = (
        left_wrist.visibility > 0.5
        and right_wrist.visibility > 0.5
        and left_shoulder.visibility > 0.5
        and right_shoulder.visibility > 0.5
    )
    if not visibility_ok:
        return False

    return left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y


def tongue_out(face_landmarks, frame):
    # FaceMesh landmark ids around mouth.
    upper_lip_id = 13
    lower_lip_id = 14
    left_corner_id = 61
    right_corner_id = 291

    lm = face_landmarks.landmark
    upper_lip = lm[upper_lip_id]
    lower_lip = lm[lower_lip_id]
    left_corner = lm[left_corner_id]
    right_corner = lm[right_corner_id]

    def as_float(value):
        return float(np.asarray(value).reshape(-1)[0])

    upper_lip_x, upper_lip_y = as_float(upper_lip.x), as_float(upper_lip.y)
    lower_lip_x, lower_lip_y = as_float(lower_lip.x), as_float(lower_lip.y)
    left_corner_x, left_corner_y = as_float(left_corner.x), as_float(left_corner.y)
    right_corner_x, right_corner_y = as_float(right_corner.x), as_float(right_corner.y)

    mouth_width = abs(right_corner_x - left_corner_x)
    if mouth_width < 1e-6:
        return False

    mouth_open_ratio = abs(lower_lip_y - upper_lip_y) / mouth_width
    if mouth_open_ratio < 0.22:
        return False

    h, w = frame.shape[:2]
    x_coords = [left_corner_x, right_corner_x, upper_lip_x, lower_lip_x]
    y_coords = [left_corner_y, right_corner_y, upper_lip_y, lower_lip_y]

    x1 = max(0, int(min(x_coords) * w) - 12)
    x2 = min(w, int(max(x_coords) * w) + 12)
    y1 = max(0, int(min(y_coords) * h) - 10)
    y2 = min(h, int(max(y_coords) * h) + 22)
    if x2 <= x1 or y2 <= y1:
        return False

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 50, 50], dtype=np.uint8)
    upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([165, 50, 50], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(
        hsv, lower_red_2, upper_red_2
    )

    red_ratio = cv2.countNonZero(mask) / float(mask.size)
    return red_ratio > 0.10


def play_sound_async(path):
    if not path or not os.path.exists(path):
        return

    system = platform.system()

    def _play():
        try:
            if system == "Darwin":
                subprocess.Popen(
                    ["afplay", path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return

            if system == "Linux":
                for cmd in ("paplay", "aplay"):
                    if shutil.which(cmd):
                        subprocess.Popen(
                            [cmd, path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        return
                return

            if system == "Windows":
                import winsound  # type: ignore

                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            return

    threading.Thread(target=_play, daemon=True).start()


def write_wav(path, samples, sample_rate=22050):
    audio = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


def create_explosion_sound(path):
    sample_rate = 22050
    duration = 1.2
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    noise = np.random.uniform(-1.0, 1.0, size=t.shape)
    env = np.exp(-4.0 * t)
    low_rumble = np.sin(2 * np.pi * 55 * t) * np.exp(-3.0 * t)
    crack = 0.6 * np.sign(np.sin(2 * np.pi * 150 * t)) * np.exp(-14.0 * t)
    explosion = (0.85 * noise * env) + (0.5 * low_rumble) + crack
    write_wav(path, explosion / (np.max(np.abs(explosion)) + 1e-6), sample_rate)


def create_scream_sound(path):
    sample_rate = 22050
    duration = 1.6
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    base_freq = 450 + 650 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.2 * t))
    phase = 2 * np.pi * np.cumsum(base_freq) / sample_rate
    tone = np.sin(phase)
    harmonics = 0.35 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    tremolo = 0.6 + 0.4 * np.sin(2 * np.pi * 9 * t)
    env = 0.25 + 0.75 * np.minimum(t / 0.2, 1.0) * np.exp(-0.2 * t)
    scream = (tone + harmonics) * tremolo * env
    write_wav(path, scream / (np.max(np.abs(scream)) + 1e-6), sample_rate)


def ensure_default_sounds(cinema_path, freaky_path):
    if cinema_path and not os.path.exists(cinema_path):
        create_explosion_sound(cinema_path)
    if freaky_path and not os.path.exists(freaky_path):
        create_scream_sound(freaky_path)


def main():
    ensure_runtime_python()
    args = parse_args()
    mp = import_real_mediapipe()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image path does not exist: {args.image}")
    if not os.path.exists(args.freaky_image):
        raise FileNotFoundError(f"Freaky image path does not exist: {args.freaky_image}")

    selected_image = load_image(args.image)
    freaky_image = load_image(args.freaky_image)
    ensure_default_sounds(args.cinema_sound, args.freaky_sound)
    cinema_sound_path = args.cinema_sound if args.cinema_sound and os.path.exists(args.cinema_sound) else ""
    freaky_sound_path = args.freaky_sound if args.freaky_sound and os.path.exists(args.freaky_sound) else ""

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils

    prev_gray = None
    cinema_count = 0
    freaky_count = 0
    last_displayed = "none"

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_results = pose.process(rgb)
            face_results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            motion_detected = False
            changed_pixels = 0
            if prev_gray is not None:
                motion_detected, changed_pixels = detect_motion(
                    prev_gray, gray, args.motion_threshold
                )
            prev_gray = gray

            hands_raised = False
            if pose_results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
                hands_raised = hands_up(pose_results.pose_landmarks, mp_pose)

            tongue_detected = False
            if face_results.multi_face_landmarks:
                tongue_detected = tongue_out(face_results.multi_face_landmarks[0], frame)

            status = (
                f"Motion: {'YES' if motion_detected else 'NO'} | "
                f"Hands up: {'YES' if hands_raised else 'NO'} | "
                f"Tongue: {'YES' if tongue_detected else 'NO'} | px: {changed_pixels}"
            )
            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Cinema count: {cinema_count}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 0),
                5,
            )
            cv2.putText(
                frame,
                f"Cinema count: {cinema_count}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 200, 0),
                3,
            )
            cv2.putText(
                frame,
                f"Freaky count: {freaky_count}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 0),
                5,
            )
            cv2.putText(
                frame,
                f"Freaky count: {freaky_count}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 100, 255),
                3,
            )

            cv2.imshow("Camera", frame)

            display_now = "none"
            if hands_raised and tongue_detected:
                display_now = "freaky"
                resized = cv2.resize(freaky_image, (frame.shape[1], frame.shape[0]))
                cv2.imshow("Selected Image", resized)
            elif motion_detected and hands_raised:
                display_now = "cinema"
                resized = cv2.resize(selected_image, (frame.shape[1], frame.shape[0]))
                cv2.imshow("Selected Image", resized)
            else:
                cv2.imshow("Selected Image", np.zeros_like(frame))

            if display_now != last_displayed:
                if display_now == "cinema":
                    cinema_count += 1
                    play_sound_async(cinema_sound_path)
                elif display_now == "freaky":
                    freaky_count += 1
                    play_sound_async(freaky_sound_path)
            last_displayed = display_now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()