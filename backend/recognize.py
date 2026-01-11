import cv2
import numpy as np
import os

# ขนาดภาพใบหน้า
FACE_SIZE = (120, 120)

# Cache face cascade
_face_cascade = None

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade

def detect_face(img_bgr):
    """
    ตรวจจับใบหน้าอย่างง่าย
    Return: (face_crop, rect) หรือ (None, None)
    """
    if img_bgr is None:
        return None, None
    
    cascade = get_face_cascade()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # ตรวจจับใบหน้า - ใช้ค่าที่ยืดหยุ่น
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(50, 50)
    )
    
    if len(faces) == 0:
        return None, None
    
    # เอาใบหน้าที่ใหญ่ที่สุด
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    
    # Crop และ resize
    face_crop = gray[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, FACE_SIZE)
    
    return face_crop, (x, y, w, h)

def load_faces_from_db(db_rows):
    """
    โหลดภาพใบหน้าจาก DB
    """
    images = []
    labels = []
    
    for r in db_rows:
        path = r["image_path"]
        student_pk = int(r["student_pk"])
        
        if not os.path.exists(path):
            continue
            
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Resize ให้เท่ากัน
        if img.shape != FACE_SIZE:
            img = cv2.resize(img, FACE_SIZE)
            
        images.append(img)
        labels.append(student_pk)
    
    return images, labels

# LBPH Model Cache
_recognizer = None
_last_count = 0

def get_recognizer(images, labels):
    """
    สร้างหรือใช้ cached recognizer
    """
    global _recognizer, _last_count
    
    if len(images) == 0:
        return None
    
    # ถ้าจำนวนภาพเปลี่ยน ต้อง train ใหม่
    if _recognizer is None or _last_count != len(images):
        _recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        _recognizer.train(images, np.array(labels, dtype=np.int32))
        _last_count = len(images)
    
    return _recognizer

def invalidate_recognizer_cache():
    """เรียกเมื่อมีการเพิ่ม/ลบใบหน้า"""
    global _recognizer, _last_count
    _recognizer = None
    _last_count = 0

def recognize_face(face_crop, images, labels, threshold=100):
    """
    จดจำใบหน้า
    Return: (student_pk, confidence) หรือ (None, 0)
    threshold: ยิ่งสูงยิ่งง่าย (100 = ง่ายมาก)
    """
    recognizer = get_recognizer(images, labels)
    if recognizer is None:
        return None, 0
    
    pred_label, conf = recognizer.predict(face_crop)
    
    # conf ยิ่งต่ำยิ่งดี
    if conf <= threshold:
        return pred_label, conf
    
    return None, conf

# ========== Legacy functions for compatibility ==========

def get_best_face(img_bgr, brutal=False):
    """Compatibility wrapper"""
    face_crop, rect = detect_face(img_bgr)
    if face_crop is None:
        return None
    
    return {
        "rotated_img": img_bgr,
        "rect": rect,
        "face_crop": face_crop,
        "quality": {"blur": 100, "brightness": 128, "ok_sharp": True, "ok_bright": True, "ok_size": True},
        "ok": True
    }

def pick_best_quality_frame(fetch_frame_fn, attempts=5, brutal=True, sleep=0.05):
    """Compatibility wrapper"""
    import time
    for _ in range(attempts):
        try:
            img = fetch_frame_fn()
            res = get_best_face(img, brutal=brutal)
            if res is not None:
                return res
        except:
            pass
        time.sleep(sleep)
    return None

def load_training_data(db_rows):
    """Compatibility wrapper"""
    return load_faces_from_db(db_rows)

def train_lbph(images, labels):
    """Compatibility wrapper"""
    return get_recognizer(images, labels)

def lbph_score_from_confidence(conf):
    """แปลง confidence เป็น score 0-1"""
    return max(0, min(1, 1 - (conf / 150)))
