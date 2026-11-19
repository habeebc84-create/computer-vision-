"""CSV + PDF report generation from real detection-history records (reportlab)."""
import csv, io
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def build_csv(records, dest: Path):
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Date/Time", "Input Type", "Model", "Classes", "Object Count",
                    "Avg Confidence", "FPS", "Inference ms", "Output"])
        for r in records:
            w.writerow([r["id"], r["created_at"], r["input_type"], r.get("model", ""),
                        "; ".join(r["classes"]), r["object_count"], r["avg_conf"],
                        r["fps"], r["inference_ms"], r.get("output_path", "")])
    return dest

def build_pdf(records, dest: Path, title="Real-Time Object Detection — Report"):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(dest), pagesize=A4, title=title)
    story = [Paragraph("Real-Time Object Detection System", styles["Title"]),
             Paragraph("YOLOv8 &amp; Deep Learning — ITA0523 Computer Vision", styles["Normal"]),
             Spacer(1, 6*mm),
             Paragraph(f"Records in this report: {len(records)}", styles["Normal"]),
             Spacer(1, 4*mm)]
    data = [["ID", "Date/Time", "Type", "Classes", "Count", "Avg Conf", "FPS"]]
    for r in records:
        data.append([str(r["id"]), r["created_at"], r["input_type"],
                     ", ".join(r["classes"])[:40], str(r["object_count"]),
                     f"{r['avg_conf']:.2f}", f"{r['fps']:.1f}"])
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#334155")),
                           ("FONTSIZE", (0, 0), (-1, -1), 8),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]))
    story.append(t)
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("Generated from recorded detection history only — no simulated data.", styles["Normal"]))
    doc.build(story)
    return dest
