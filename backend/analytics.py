"""SQLite detection-history store + analytics computed only from real records."""
import json, sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB = BASE / "database" / "detections.db"

@contextmanager
def conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()

def init_db():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS detections(
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            input_type TEXT NOT NULL, classes TEXT NOT NULL DEFAULT '[]',
            object_count INTEGER NOT NULL DEFAULT 0, avg_conf REAL DEFAULT 0,
            fps REAL DEFAULT 0, inference_ms REAL DEFAULT 0,
            output_path TEXT DEFAULT '', model TEXT DEFAULT '')""")

def record(input_type, detections, fps=0.0, inference_ms=0.0, output_path="", model=""):
    init_db()
    classes = sorted({d["class_name"] for d in detections})
    avg = round(sum(d["confidence"] for d in detections)/len(detections), 4) if detections else 0.0
    import datetime
    with conn() as c:
        cur = c.execute("INSERT INTO detections(created_at,input_type,classes,object_count,avg_conf,fps,inference_ms,output_path,model) VALUES(?,?,?,?,?,?,?,?,?)",
            (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), input_type, json.dumps(classes),
             len(detections), avg, round(float(fps), 2), round(float(inference_ms), 1), output_path, model))
        return cur.lastrowid

def record_summary(input_type, object_count, classes, fps=0.0, inference_ms=0.0,
                   output_path="", model="", avg_conf=0.0):
    """Record a session whose per-object list is not retained (e.g. video totals)."""
    init_db()
    import datetime
    with conn() as c:
        cur = c.execute("INSERT INTO detections(created_at,input_type,classes,object_count,avg_conf,fps,inference_ms,output_path,model) VALUES(?,?,?,?,?,?,?,?,?)",
            (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), input_type, json.dumps(sorted(classes)),
             int(object_count), round(float(avg_conf), 4), round(float(fps), 2),
             round(float(inference_ms), 1), output_path, model))
        return cur.lastrowid

def list_history(search="", input_type="", date_from="", date_to="", limit=200):
    init_db()
    q = "SELECT * FROM detections WHERE 1=1"; args = []
    if input_type: q += " AND input_type=?"; args.append(input_type)
    if date_from: q += " AND date(created_at)>=date(?)"; args.append(date_from)
    if date_to: q += " AND date(created_at)<=date(?)"; args.append(date_to)
    if search: q += " AND (classes LIKE ? OR input_type LIKE ?)"; args += [f"%{search}%", f"%{search}%"]
    q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
    with conn() as c:
        rows = [dict(r) for r in c.execute(q, args).fetchall()]
    for r in rows: r["classes"] = json.loads(r["classes"])
    return rows

def get_one(rid):
    init_db()
    with conn() as c:
        r = c.execute("SELECT * FROM detections WHERE id=?", (rid,)).fetchone()
    if not r: return None
    d = dict(r); d["classes"] = json.loads(d["classes"]); return d

def delete_one(rid):
    init_db()
    with conn() as c:
        c.execute("DELETE FROM detections WHERE id=?", (rid,))

def summary():
    init_db()
    with conn() as c:
        total = c.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        by_type = {r[0]: r[1] for r in c.execute("SELECT input_type, COUNT(*) FROM detections GROUP BY input_type")}
        objects = c.execute("SELECT COALESCE(SUM(object_count),0) FROM detections").fetchone()[0]
        avg_conf = c.execute("SELECT COALESCE(AVG(avg_conf),0) FROM detections").fetchone()[0]
        avg_fps = c.execute("SELECT COALESCE(AVG(fps),0) FROM detections WHERE fps>0").fetchone()[0]
        all_classes = set()
        for (cl,) in c.execute("SELECT classes FROM detections"):
            all_classes.update(json.loads(cl))
        recent = [dict(r) for r in c.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 8")]
        for r in recent: r["classes"] = json.loads(r["classes"])
    return {"total_detections": int(objects), "total_records": total,
            "images": by_type.get("image", 0), "videos": by_type.get("video", 0),
            "live": by_type.get("live", 0), "num_classes": len(all_classes),
            "avg_confidence": round(float(avg_conf), 4), "avg_fps": round(float(avg_fps), 2),
            "recent": recent}

def class_distribution():
    init_db()
    cnt = Counter()
    with conn() as c:
        for (cl,) in c.execute("SELECT classes FROM detections"):
            cnt.update(json.loads(cl))
    return [{"name": k, "count": v} for k, v in cnt.most_common()]

def activity(days=14):
    init_db()
    with conn() as c:
        rows = c.execute("SELECT date(created_at) d, COUNT(*) n, COALESCE(SUM(object_count),0) o FROM detections GROUP BY d ORDER BY d DESC LIMIT ?", (days,)).fetchall()
    return [{"date": r[0], "sessions": r[1], "objects": r[2]} for r in reversed(rows)]

def confidence_buckets():
    init_db()
    buckets = {"0.25–0.5": 0, "0.5–0.7": 0, "0.7–0.85": 0, "0.85–1.0": 0}
    with conn() as c:
        for (v,) in c.execute("SELECT avg_conf FROM detections"):
            if v < 0.5: buckets["0.25–0.5"] += 1
            elif v < 0.7: buckets["0.5–0.7"] += 1
            elif v < 0.85: buckets["0.7–0.85"] += 1
            else: buckets["0.85–1.0"] += 1
    return [{"range": k, "count": v} for k, v in buckets.items()]
