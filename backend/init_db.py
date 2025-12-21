from database import get_conn


BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS students (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS student_faces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  student_pk INTEGER NOT NULL,
  image_path TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
  FOREIGN KEY (student_pk) REFERENCES students(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_student_faces_student_pk
ON student_faces(student_pk);

-- ✅ สร้างคาบ (session) เพื่อแยกการเช็คชื่อเป็นแต่ละคาบ
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  start_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
  created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_created_at
ON sessions(created_at);

-- บันทึกเฉพาะกรณีที่ match เท่านั้น (ผูกกับ session)
CREATE TABLE IF NOT EXISTS attendance_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  student_pk INTEGER NOT NULL,
  student_id TEXT NOT NULL,
  name TEXT NOT NULL,
  checked_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
  source TEXT NOT NULL DEFAULT 'mobile_cam',
  FOREIGN KEY (student_pk) REFERENCES students(id),
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- ห้ามเช็คซ้ำ "ในคาบเดียวกัน"
CREATE UNIQUE INDEX IF NOT EXISTS uq_attendance_session_student
ON attendance_logs(session_id, student_pk);

CREATE INDEX IF NOT EXISTS idx_attendance_session_id
ON attendance_logs(session_id);

CREATE INDEX IF NOT EXISTS idx_attendance_student_pk
ON attendance_logs(student_pk);

CREATE INDEX IF NOT EXISTS idx_attendance_checked_at
ON attendance_logs(checked_at);
"""


def _column_exists(cur, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] if not isinstance(r, dict) else r["name"] for r in cur.fetchall()]
    return col in cols


def init_db():
    """สร้าง schema + ทำ migration แบบปลอดภัย

    ✅ รุ่นก่อนหน้าไม่มี sessions และ attendance_logs ไม่มี session_id
    - ถ้าพบ schema เก่า -> เพิ่ม sessions + เพิ่มคอลัมน์ session_id แล้ว backfill ให้เป็นคาบ Legacy
    """

    conn = get_conn()
    cur = conn.cursor()

    # 1) ตารางหลัก (students, student_faces) - มีอยู่แล้วก็ไม่กระทบ
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS students (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          student_id TEXT NOT NULL UNIQUE,
          name TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS student_faces (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          student_pk INTEGER NOT NULL,
          image_path TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
          FOREIGN KEY (student_pk) REFERENCES students(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_student_faces_student_pk
        ON student_faces(student_pk);
        """
    )

    # 2) sessions (ใหม่)
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          title TEXT NOT NULL,
          start_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
          created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_created_at
        ON sessions(created_at);
        """
    )

    # 3) attendance_logs อาจเป็น schema เก่า -> ตรวจสอบแล้ว migrate
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='attendance_logs'"
    )
    has_att = cur.fetchone() is not None

    if not has_att:
        # ไม่มี -> สร้างแบบใหม่
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS attendance_logs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id INTEGER NOT NULL,
              student_pk INTEGER NOT NULL,
              student_id TEXT NOT NULL,
              name TEXT NOT NULL,
              checked_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
              source TEXT NOT NULL DEFAULT 'mobile_cam',
              FOREIGN KEY (student_pk) REFERENCES students(id),
              FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            """
        )
    else:
        # มีแล้ว -> ถ้าเป็น schema เก่า ให้เพิ่ม session_id แล้ว backfill
        if not _column_exists(cur, "attendance_logs", "session_id"):
            # สร้างคาบ Legacy ไว้รองรับข้อมูลเก่า
            cur.execute("SELECT id FROM sessions ORDER BY id ASC LIMIT 1")
            row = cur.fetchone()
            legacy_id = row[0] if row else None
            if legacy_id is None:
                cur.execute(
                    "INSERT INTO sessions(title) VALUES (?)",
                    ("Legacy (ก่อนแยกคาบ)",),
                )
                legacy_id = cur.lastrowid

            # เพิ่มคอลัมน์แบบ nullable ก่อน
            cur.execute("ALTER TABLE attendance_logs ADD COLUMN session_id INTEGER")

            # backfill ข้อมูลเดิมให้ไปอยู่คาบ legacy
            cur.execute(
                "UPDATE attendance_logs SET session_id = ? WHERE session_id IS NULL",
                (int(legacy_id),),
            )

    # 4) indexes / constraints (สร้างซ้ำได้ เพราะ IF NOT EXISTS)
    cur.executescript(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_attendance_session_student
        ON attendance_logs(session_id, student_pk);

        CREATE INDEX IF NOT EXISTS idx_attendance_session_id
        ON attendance_logs(session_id);

        CREATE INDEX IF NOT EXISTS idx_attendance_student_pk
        ON attendance_logs(student_pk);

        CREATE INDEX IF NOT EXISTS idx_attendance_checked_at
        ON attendance_logs(checked_at);
        """
    )

    # 5) ถ้ายังไม่มีคาบเลย (fresh install) -> สร้างคาบเริ่มต้นให้
    cur.execute("SELECT COUNT(*) AS c FROM sessions")
    c = cur.fetchone()[0]
    if c == 0:
        cur.execute("INSERT INTO sessions(title) VALUES (?)", ("คาบที่ 1",))

    conn.commit()
    conn.close()
