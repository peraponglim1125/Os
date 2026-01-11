import cv2
import numpy as np
import os
import time

# ====== Pi 3 Optimized Settings ======
FACE_SIZE = (100, 100)  # ลดจาก 160 เพื่อประหยัด RAM และเร็วขึ้น

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

# Cache face cascade (โหลดครั้งเดียว ไม่โหลดซ้ำ)
_face_cascade = None

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade

def get_best_face(img_bgr, brutal=False):
    """
    คืนค่า dict:
    - rotated_img (BGR)
    - rect (x,y,w,h) ของ "หน้าเดียวที่ดีที่สุด"
    - face_crop (Gray 100x100)
    - quality: blur/brightness flags
    - ok: ผ่านคุณภาพหรือยัง
    
    Optimized สำหรับ Raspberry Pi 3
    """
    if img_bgr is None:
        return None

    face_cascade = get_face_cascade()

    best = None
    best_area = 0

    # โหมดโหด: เข้มขึ้น (กรองลายไม้/ผนัง)
    if brutal:
        minNeighbors = 5
        minSize = (70, 70)
        scaleFactor = 1.15  # เร็วกว่า 1.1
    else:
        minNeighbors = 4
        minSize = (60, 60)
        scaleFactor = 1.2

    # Pi3: ลดการหมุนภาพ - กล้องตั้งตรงอยู่แล้ว
    angles_to_try = [0] if not brutal else [0, -90]

    for angle in angles_to_try:
        rotated = rotate_image(img_bgr, angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize
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

            # เกณฑ์คุณภาพ (ปรับให้ผ่านง่ายขึ้นสำหรับ Pi)
            ok_sharp = blur_score >= 12.0
            ok_bright = 15.0 <= bright <= 250.0
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

def pick_best_quality_frame(fetch_frame_fn, attempts=8, brutal=True, sleep=0.03):
    """
    ดึงหลายเฟรมเพื่อหาเฟรมที่ "ชัดที่สุด"
    - ถ้าเจอเฟรมที่ ok แล้ว จะหยุดทันที
    - Pi 3: ลด attempts จาก 15 เป็น 8
    """
    best = None
    best_blur = -1.0

    for _ in range(attempts):
        try:
            img = fetch_frame_fn()
        except:
            time.sleep(sleep)
            continue
            
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
    return images(gray 100x100), labels(student_pk)
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

# ====== LBPH Model Cache สำหรับ Pi 3 ======
_cached_recognizer = None
_cached_image_count = 0
_cache_dirty = True

def invalidate_recognizer_cache():
    """เรียกเมื่อมีการเพิ่ม/ลบใบหน้า"""
    global _cache_dirty
    _cache_dirty = True

def get_or_train_lbph(images, labels):
    """
    ใช้ cached model ถ้ามี และจำนวนภาพเท่าเดิม
    จะ train ใหม่เฉพาะเมื่อมีการเปลี่ยนแปลง
    """
    global _cached_recognizer, _cached_image_count, _cache_dirty
    
    if len(images) == 0:
        return None
    
    # ถ้า cache ยังใช้ได้ ไม่ต้อง train ใหม่
    if not _cache_dirty and _cached_recognizer is not None and _cached_image_count == len(images):
        return _cached_recognizer
    
    # Train ใหม่ - คง grid 8x8 เพื่อความแม่นยำ
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(images, np.array(labels, dtype=np.int32))
    
    # Cache ไว้
    _cached_recognizer = recognizer
    _cached_image_count = len(images)
    _cache_dirty = False
    
    return recognizer

def train_lbph(images, labels):
    """
    Wrapper สำหรับ backward compatibility
    ใช้งานผ่าน cache
    """
    return get_or_train_lbph(images, labels)

def lbph_score_from_confidence(conf):
    """
    LBPH confidence เป็น distance (ยิ่งน้อยยิ่งดี)
    แปลงเป็น score 0..1 (ยิ่งมากยิ่งดี)
    """
    score = 1.0 - min(max(conf / 120.0, 0.0), 1.0)
    return float(score)
