from flask import Flask, Response, send_file, request, jsonify
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_PATH = os.path.join(BASE_DIR, "frontend")

# =========================
# MODELS
# =========================
manual_model = YOLO(os.path.join(BASE_DIR, "runs/detect/train1/weights/best.pt"))
damage_model = YOLO(os.path.join(BASE_DIR, "runs/detect/train2/weights/best.pt"))
auto_model = YOLO(os.path.join(BASE_DIR, "runs/detect/train3/weights/best.pt"))

# =========================
# CAMERA
# =========================
IP_CAMERA_URL = "http://10.193.253.65:8080/video"
cap = cv2.VideoCapture(IP_CAMERA_URL)

# =========================
# SETTINGS
# =========================
MM_PER_PIXEL = 0.125
TOLERANCE = 5

AUTO_VALUES = {
    "BOLT_A": {"width": 23, "height": 30},
    "BOLT_B": {"width": 23, "height": 73},
    "BOLT_C": {"width": 15, "height": 43},
    "BOLT_D": {"width": 23, "height": 48}
}

# 🔥 DEFAULT MODE = AUTO
current_mode = "auto"

latest_dimensions = {"width": 0, "height": 0}
latest_result = {"status": "WAITING", "type": "-"}

# =========================
# MODE SWITCH
# =========================
@app.route('/set_mode/<mode>')
def set_mode(mode):
    global current_mode
    current_mode = mode
    return jsonify({"mode": mode})

# =========================
# MEASUREMENT FUNCTION
# =========================
def measure_object(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0, None

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)

    (_, _), (w, h), _ = rect

    width_px = min(w, h)
    height_px = max(w, h)

    return width_px, height_px, rect

# =========================
# DAMAGE CHECK FUNCTION
# =========================
def check_damage(frame):
    dmg_res = damage_model(frame, conf=0.4)[0]

    if len(dmg_res.boxes) > 0:
        defect_id = int(dmg_res.boxes.cls[0])
        defect_name = dmg_res.names[defect_id]
        return True, defect_name, dmg_res.boxes

    return False, None, None

# =========================
# VIDEO STREAM
# =========================
def generate_frames():
    global latest_dimensions, latest_result

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        annotated = frame.copy()

        # =========================
        # 🔥 STEP 1: DAMAGE CHECK
        # =========================
        is_damaged, defect_name, dmg_boxes = check_damage(frame)

        if is_damaged:
            latest_result = {"status": "DAMAGED", "type": defect_name.upper()}

            for box in dmg_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(annotated, defect_name.upper(),
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255), 3)

            cv2.putText(annotated, "DAMAGED",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 4)

        else:
            # =========================
            # AUTO MODE
            # =========================
            if current_mode == "auto":
                results = auto_model(frame, conf=0.7)

                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    raw_label = auto_model.names[cls_id]

                    label = raw_label.replace("-", "_").replace(" ", "_").upper()
                    conf = float(box.conf[0]) * 100

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    w_px, h_px, _ = measure_object(roi)

                    width_mm = round(w_px * MM_PER_PIXEL, 2)
                    height_mm = round(h_px * MM_PER_PIXEL, 2)

                    latest_dimensions["width"] = width_mm
                    latest_dimensions["height"] = height_mm

                    result = "UNKNOWN"

                    if label in AUTO_VALUES:
                        std = AUTO_VALUES[label]

                        if (abs(std["width"] - width_mm) <= TOLERANCE and
                            abs(std["height"] - height_mm) <= TOLERANCE):
                            result = "PASS"
                        else:
                            result = "FAIL"

                    latest_result = {"status": result, "type": label}

                    color = (0,255,0) if result == "PASS" else (0,0,255)

                    cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 4)

                    cv2.putText(annotated, label, (x1, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                    cv2.putText(annotated, f"{conf:.1f}%",
                                (x2-120, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                    cv2.putText(annotated,
                                f"W:{width_mm}  H:{height_mm}",
                                (x1, y2+40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

            # =========================
            # MANUAL MODE
            # =========================
            else:
                results = manual_model(frame, conf=0.6)

                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = manual_model.names[cls_id]
                    conf = float(box.conf[0]) * 100

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    w_px, h_px, _ = measure_object(roi)

                    width_mm = round(w_px * MM_PER_PIXEL, 2)
                    height_mm = round(h_px * MM_PER_PIXEL, 2)

                    latest_dimensions["width"] = width_mm
                    latest_dimensions["height"] = height_mm

                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 4)

                    cv2.putText(annotated, label.upper(), (x1, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                    cv2.putText(annotated, f"{conf:.1f}%",
                                (x2-120, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                    cv2.putText(annotated,
                                f"W:{width_mm}  H:{height_mm}",
                                (x1, y2+40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)

        # =========================
        # STREAM OUTPUT
        # =========================
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return send_file(os.path.join(FRONTEND_PATH, "index.html"))

@app.route('/style.css')
def style():
    return send_file(os.path.join(FRONTEND_PATH, "style.css"))

@app.route('/script.js')
def script():
    return send_file(os.path.join(FRONTEND_PATH, "script.js"))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    return jsonify(latest_result)

@app.route('/check', methods=['POST'])
def check():
    data = request.json
    dims = data.get("dimensions", {})

    result = "PASS"

    if "Width" in dims:
        if abs(dims["Width"] - latest_dimensions["width"]) > TOLERANCE:
            result = "FAIL"

    if "Height" in dims:
        if abs(dims["Height"] - latest_dimensions["height"]) > TOLERANCE:
            result = "FAIL"

    return jsonify({"status": result})


if __name__ == "__main__":
    app.run(debug=True)