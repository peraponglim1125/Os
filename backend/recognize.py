import cv2
import numpy as np
import os
import time

FACE_SIZE = (160, 160)

def rotate_image(img_bgr, angle):
    if angle == 0:
        return img_bgr
    if angle == -90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_bgr

def laplacian_var(gray):
    # วัดความชัด (ยิ่งมากยิ่งชัด)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def brightness_mean(gray):
    return float(np.mean(gray))

def get_best_face(img_bgr, brutal=False):
    """
    คืนค่า dict:
    - rotated_img (BGR)
    - rect (x,y,w,h) ของ "หน้าเดียวที่ดีที่สุด"
    - face_crop (Gray 160x160)
    - quality: blur/brightness flags
    - ok: ผ่านคุณภาพหรือยัง
    """
    if img_bgr is None:
        return None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    best = None
    best_area = 0

    # โหมดโหด: เข้มขึ้น (กรองลายไม้/ผนัง)
    if brutal:
        minNeighbors = 6
        minSize = (90, 90)
    else:
        minNeighbors = 5
        minSize = (80, 80)

    for angle in [0, -90, 90]:
        rotated = rotate_image(img_bgr, angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=minNeighbors, minSize=minSize
        )
        if len(faces) == 0:
            continue

        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        area = w * h
        if area > best_area:
            crop = gray[y:y+h, x:x+w]
            crop = cv2.resize(crop, FACE_SIZE)
            best_area = area

            blur_score = laplacian_var(crop)
            bright = brightness_mean(crop)

            # เกณฑ์คุณภาพ (ปรับให้ผ่านง่ายขึ้น)
            ok_sharp = blur_score >= 15.0
            ok_bright = 20.0 <= bright <= 250.0
            ok_size = (w >= minSize[0] and h >= minSize[1])
            ok = ok_sharp and ok_bright and ok_size

            best = {
                "rotated_img": rotated,
                "rect": (int(x), int(y), int(w), int(h)),
                "face_crop": crop,
                "quality": {
                    "blur": blur_score,
                    "brightness": bright,
                    "ok_sharp": ok_sharp,
                    "ok_bright": ok_bright,
                    "ok_size": ok_size,
                },
                "ok": ok,
            }

    return best

def crop_face(img_bgr):
    res = get_best_face(img_bgr, brutal=True)
    if res:
        return res["face_crop"]
    return None

def pick_best_quality_frame(fetch_frame_fn, attempts=15, brutal=True, sleep=0.05):
    """
    ดึงหลายเฟรมเพื่อหาเฟรมที่ "ชัดที่สุด"
    - ถ้าเจอเฟรมที่ ok แล้ว จะหยุดทันที
    """
    best = None
    best_blur = -1.0

    for _ in range(attempts):
        img = fetch_frame_fn()
        res = get_best_face(img, brutal=brutal)
        if res is None:
            time.sleep(sleep)
            continue

        blur = res["quality"]["blur"]
        if blur > best_blur:
            best_blur = blur
            best = res

        if res["ok"]:
            return res

        time.sleep(sleep)

    return best

def load_training_data(db_rows):
    """
    db_rows: rows ที่มี student_pk, image_path
    return images(gray160), labels(student_pk)
    """
    images = []
    labels = []
    for r in db_rows:
        path = r["image_path"]
        if not os.path.exists(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != FACE_SIZE:
            img = cv2.resize(img, FACE_SIZE)
        images.append(img)
        labels.append(int(r["student_pk"]))
    return images, labels

def train_lbph(images, labels):
    """
    LBPH: เหมาะกับ Raspberry Pi (เร็ว)
    """
    if len(images) == 0:
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(images, np.array(labels, dtype=np.int32))
    return recognizer

def lbph_score_from_confidence(conf):
    """
    LBPH confidence เป็น distance (ยิ่งน้อยยิ่งดี)
    แปลงเป็น score 0..1 (ยิ่งมากยิ่งดี)
    """
    score = 1.0 - min(max(conf / 120.0, 0.0), 1.0)
    return float(score)
