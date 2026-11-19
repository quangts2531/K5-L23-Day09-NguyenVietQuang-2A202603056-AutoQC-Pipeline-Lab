"""Tạo dashboard HTML độc lập từ các báo cáo của pipeline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def read_label_boxes(path: Path, classes: list[str]) -> list[dict]:
    boxes = []
    if not path.exists():
        return boxes
    for line_no, text in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        try:
            parts = text.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            cx, cy, width, height = map(float, parts[1:])
            if not all(math.isfinite(value) for value in (cx, cy, width, height)) or width <= 0 or height <= 0:
                continue
        except (ValueError, OverflowError):
            continue
        boxes.append({
            "class_id": class_id,
            "class_name": classes[class_id] if 0 <= class_id < len(classes) else f"class {class_id}",
            "cx": cx, "cy": cy, "w": width, "h": height, "line": line_no,
        })
    return boxes


def phase_summary(rows: list[dict[str, str]]) -> dict:
    return {
        "rows": len(rows),
        "images": len({row.get("image", "") for row in rows if row.get("image")}),
        "errors": sum(row.get("severity", "").lower() == "error" for row in rows),
        "warnings": sum(row.get("severity", "").lower() == "warn" for row in rows),
        "codes": Counter(row.get("code", "(không có mã)") for row in rows),
    }


def code_table(counter: Counter, *, limit: int = 8) -> str:
    if not counter:
        return '<p class="empty">Không có lỗi hoặc cảnh báo ở bước này.</p>'
    rows = "".join(
        f"<tr><td><code>{escape(code)}</code></td><td class=\"num\">{count}</td></tr>"
        for code, count in counter.most_common(limit)
    )
    return f'<table><thead><tr><th>Mã</th><th class="num">Số dòng</th></tr></thead><tbody>{rows}</tbody></table>'


def build_html(reports_dir: Path) -> str:
    reports_dir = reports_dir.resolve()
    summary = read_json(reports_dir / "08_summary.json")
    verdict = summary.get("verdict") or {}
    decision = str(verdict.get("decision", "Chưa có kết luận"))
    v1_rows = read_csv(reports_dir / "02_error_list_v1.csv")
    v2_rows = read_csv(reports_dir / "04_error_list_v2.csv")
    v3_rows = read_csv(reports_dir / "07_error_list_v3.csv")
    intake_rows = read_csv(reports_dir / "01_intake.csv")
    fixes = read_csv(reports_dir / "03_fixes.csv")
    queue = read_csv(reports_dir / "05_review_queue.csv")
    review_log = read_csv(reports_dir / "06_review_log.csv")
    class_stats = read_csv(reports_dir / "class_stats.csv")
    cfg = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8")) or {}
    classes = cfg.get("classes", [])
    score_threshold = float(cfg.get("second_opinion", {}).get("conf", 0.6))
    cache_path = ROOT / cfg.get("models", {}).get("checker", {}).get("cache", "data/cache/checker_preds.json")
    cached_preds = read_json(cache_path)
    preds_by_stem = {Path(filename).stem: rows for filename, rows in cached_preds.items()} if isinstance(cached_preds, dict) else {}
    images_dir = ROOT / "work/run/ds/images"
    labels_dir = ROOT / "work/run/ds/labels"
    disk_images = {path.stem: path for path in images_dir.iterdir() if path.is_file()} if images_dir.exists() else {}
    image_meta = {}
    for stem in {row.get("image", "") for row in v3_rows}:
        image_path = disk_images.get(stem)
        model_boxes = []
        for prediction in preds_by_stem.get(stem, []):
            try:
                class_id, conf, cx, cy, width, height = prediction
                class_id = int(class_id)
                conf, cx, cy, width, height = map(float, (conf, cx, cy, width, height))
                if conf < score_threshold or not all(math.isfinite(value) for value in (conf, cx, cy, width, height)) or width <= 0 or height <= 0:
                    continue
            except (TypeError, ValueError, OverflowError):
                continue
            model_boxes.append({
                "class_id": class_id,
                "class_name": classes[class_id] if 0 <= class_id < len(classes) else f"class {class_id}",
                "conf": conf, "cx": cx, "cy": cy, "w": width, "h": height,
            })
        image_meta[stem] = {
            "src": quote(os.path.relpath(image_path, reports_dir), safe="/") if image_path else "",
            "labels": read_label_boxes(labels_dir / f"{stem}.txt", classes),
            "predictions": model_boxes,
        }
    reviewed_images = {row.get("image", "") for row in review_log if row.get("image")}
    queue_priority = {row.get("image", ""): float(row.get("priority") or 0) for row in queue if row.get("image")}
    pending_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in v3_rows:
        stem = row.get("image", "")
        if row.get("severity", "").lower() == "warn" and stem and stem not in reviewed_images:
            pending_by_image[stem].append(row)
    review_items = [
        {"image": stem, "priority": queue_priority.get(stem, 0), "issues": issues}
        for stem, issues in pending_by_image.items()
    ]
    review_items.sort(key=lambda item: (-item["priority"], item["image"]))
    phases = [phase_summary(rows) for rows in (v1_rows, v2_rows, v3_rows)]
    errors = [row for row in v3_rows if row.get("severity", "").lower() == "error"]
    warnings = [row for row in v3_rows if row.get("severity", "").lower() == "warn"]
    total_images = verdict.get("n_images", phases[2]["images"] or phases[1]["images"] or phases[0]["images"])
    warn_images = verdict.get("n_warn_images")
    if warn_images is None:
        warn_images = len({row.get("image", "") for row in warnings if row.get("image")})
    n_errors = verdict.get("n_errors", len(errors))
    reviewed = summary.get("reviewed", len(review_log))
    budget = summary.get("budget", len(queue))
    warn_ratio = verdict.get("warn_ratio")
    if warn_ratio is None:
        warn_ratio = warn_images / total_images if total_images else 0
    exported = summary.get("exported")
    report_data = {
        "issues": v3_rows,
        "images": image_meta,
        "reviewItems": review_items,
        "runKey": hashlib.sha256(
            (summary.get("config_sha1", "") + summary.get("review_sim_sha1", "") +
             ((reports_dir / "07_error_list_v3.csv").read_text(encoding="utf-8", errors="replace")
              if (reports_dir / "07_error_list_v3.csv").exists() else "")).encode("utf-8")
        ).hexdigest()[:16],
        "scoreThreshold": score_threshold,
        "queue": queue,
        "phases": phases,
        "reasons": verdict.get("reasons", []),
        "generated": datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S"),
    }
    embedded = json.dumps(report_data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    decision_class = decision.lower() if decision in {"Release", "Hold", "Rework"} else "unknown"
    reason_html = "".join(f"<li>{escape(str(reason))}</li>" for reason in verdict.get("reasons", []))
    if not reason_html:
        reason_html = "<li>Chưa có file kết luận 08_summary.json. Có thể pipeline đang tạm dừng trước bước duyệt.</li>"
    export_text = "Đã xuất dataset" if exported else "Chưa xuất dataset"
    export_detail = Path(exported).name if exported else "Pipeline không tạo bản xuất trong lần chạy này."
    ratio_pct = f"{float(warn_ratio) * 100:.2f}%"
    phase_names = ["Trước sửa (v1)", "Sau tự động sửa (v2)", "Sau duyệt (v3)"]
    phase_cards = "".join(
        f'''<article class="phase"><div class="phase-title">{name}</div>
            <div class="phase-number">{phase["images"]}<span> ảnh có ghi nhận</span></div>
            <div class="phase-meta">{phase["rows"]} dòng · {phase["errors"]} lỗi · {phase["warnings"]} cảnh báo</div>
            <details><summary>Xem mã lỗi</summary>{code_table(phase["codes"])}</details></article>'''
        for name, phase in zip(phase_names, phases)
    )
    files = [
        ("08_summary.json", "Kết luận pipeline"),
        ("01_intake.csv", "Ảnh nhận vào"),
        ("02_error_list_v1.csv", "Danh sách trước sửa"),
        ("03_fixes.csv", "Các sửa tự động"),
        ("04_error_list_v2.csv", "Danh sách sau sửa"),
        ("05_review_queue.csv", "Hàng đợi duyệt"),
        ("06_review_log.csv", "Nhật ký duyệt"),
        ("07_error_list_v3.csv", "Danh sách cuối"),
        ("07_error_list_v3.txt", "Báo cáo cuối dạng chữ"),
        ("class_stats.csv", "Thống kê lớp"),
    ]
    links = "".join(
        f'<a class="file-link" href="{escape(filename, quote=True)}">{escape(label)} <span>↗</span></a>'
        for filename, label in files if (reports_dir / filename).exists()
    )
    def select_options(values: set[str], placeholder: str) -> str:
        return f'<option value="">{escape(placeholder)}</option>' + "".join(
            f'<option value="{escape(value, quote=True)}">{escape(value)}</option>'
            for value in sorted(values)
        )

    code_options = select_options({row.get("code", "") for row in v3_rows if row.get("code")}, "Tất cả mã lỗi")
    check_options = select_options({row.get("check", "") for row in v3_rows if row.get("check")}, "Tất cả nhóm kiểm tra")
    image_options = select_options({row.get("image", "") for row in v3_rows if row.get("image")}, "Tất cả ảnh")
    final_issue_note = f"{len(v3_rows)} dòng từ báo cáo cuối; ảnh xanh là nhãn, cam là dự đoán AI (conf ≥ {score_threshold:g}). Lọc theo ảnh, mã lỗi hoặc mô tả."
    intake_count = len(intake_rows)
    fix_count = len(fixes)
    class_rows = "".join(
        f'<tr><td>{escape(row.get("class_token", ""))}</td><td class="num">{escape(row.get("boxes", ""))}</td></tr>'
        for row in class_stats
    ) or '<tr><td colspan="2" class="empty">Chưa có thống kê</td></tr>'
    queue_preview = "".join(
        f'<tr><td>{escape(row.get("image", ""))}</td><td class="num">{escape(row.get("priority", ""))}</td><td>{escape(row.get("codes", ""))}</td></tr>'
        for row in queue[:10]
    ) or '<tr><td colspan="3" class="empty">Hàng đợi trống hoặc chưa được tạo.</td></tr>'

    return f'''<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>Báo cáo QC — {escape(decision)}</title>
  <style>
    :root {{ --ink:#182230; --muted:#64748b; --line:#e2e8f0; --paper:#f4f7fb; --blue:#2563eb; --green:#15803d; --amber:#b45309; --red:#b91c1c; }}
    * {{ box-sizing:border-box }} body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
    .wrap {{ width:min(1180px,calc(100% - 36px)); margin:30px auto 56px }}
    header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:20px; margin-bottom:20px }}
    h1 {{ font-size:clamp(26px,4vw,36px); line-height:1.15; letter-spacing:-.04em; margin:3px 0 8px }}
    h2 {{ margin:0; font-size:19px; letter-spacing:-.02em }} h3 {{ font-size:16px; margin:0 0 12px }}
    p {{ margin:0 }} .eyebrow,.muted {{ color:var(--muted) }} .eyebrow {{ font-size:12px; font-weight:750; letter-spacing:.12em; text-transform:uppercase }}
    .stamp {{ color:var(--muted); font-size:13px; text-align:right; white-space:nowrap }}
    .hero,.card,.phase,.panel {{ border:1px solid var(--line); border-radius:16px; background:#fff; box-shadow:0 5px 20px #1e293b08 }}
    .hero {{ display:grid; grid-template-columns:1fr minmax(210px,280px); gap:22px; padding:24px; border-top:4px solid var(--blue); margin-bottom:16px }}
    .hero.release {{ border-top-color:var(--green) }} .hero.hold {{ border-top-color:#d97706 }} .hero.rework {{ border-top-color:var(--red) }}
    .decision {{ display:inline-flex; align-items:center; gap:8px; border-radius:999px; padding:6px 11px; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.06em; background:#eff6ff; color:var(--blue); margin-bottom:10px }}
    .decision.release {{ background:#dcfce7; color:var(--green) }} .decision.hold {{ background:#fef3c7; color:var(--amber) }} .decision.rework {{ background:#fee2e2; color:var(--red) }}
    .reasons {{ margin:14px 0 0; padding-left:20px; color:#425166 }} .reasons li+li {{ margin-top:5px }}
    .export-box {{ align-self:center; border-left:1px solid var(--line); padding-left:22px }} .export-box strong {{ display:block; font-size:17px; margin:2px 0 4px }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:16px }}
    .card {{ padding:17px 18px }} .card-label {{ color:var(--muted); font-size:12px; font-weight:700 }} .card-value {{ font-size:30px; line-height:1.2; font-weight:800; letter-spacing:-.04em; margin-top:8px }} .card-foot {{ color:var(--muted); font-size:12px; margin-top:5px }}
    .progress {{ height:7px; border-radius:9px; background:#e8edf5; overflow:hidden; margin-top:12px }} .progress i {{ display:block; height:100%; background:var(--blue); border-radius:9px }}
    .panel {{ padding:21px; margin-bottom:16px }} .panel-head {{ display:flex; justify-content:space-between; align-items:end; gap:16px; margin-bottom:16px }}
    .phases {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px }} .phase {{ padding:17px; box-shadow:none }}
    .phase-title {{ color:var(--muted); font-weight:700; font-size:13px }} .phase-number {{ font-size:26px; font-weight:800; margin-top:12px }} .phase-number span {{ font-size:13px; color:var(--muted); font-weight:500 }} .phase-meta {{ font-size:12px; color:var(--muted); margin:2px 0 12px }}
    details {{ border-top:1px solid var(--line); padding-top:10px }} summary {{ cursor:pointer; color:var(--blue); font-size:13px; font-weight:700 }}
    .two-col {{ display:grid; grid-template-columns:minmax(0,1.5fr) minmax(280px,.8fr); gap:16px }}
    .toolbar {{ display:flex; flex-wrap:wrap; align-items:center; gap:9px; margin-bottom:13px }} input,.toolbar select {{ max-width:100%; border:1px solid var(--line); border-radius:9px; padding:9px 10px; color:var(--ink); background:#fff; font:inherit; font-size:13px; outline:none }} input {{ width:min(100%,320px) }} .toolbar select {{ min-width:135px }} input:focus,.toolbar select:focus {{ border-color:#7aa5fb; box-shadow:0 0 0 3px #2563eb18 }}
    .visual-toggles {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:12px; color:#526174; font-size:12px }} .visual-toggles label {{ display:flex; align-items:center; gap:6px; cursor:pointer }} .visual-toggles input {{ width:auto; accent-color:var(--blue) }}
    .review-intro {{ border:1px solid #bfdbfe; background:#eff6ff; padding:15px 17px; border-radius:12px; color:#334155; margin-bottom:14px }} .review-intro strong {{ color:#1d4ed8 }} .review-steps {{ margin:8px 0 0; padding-left:20px; font-size:13px }} .review-steps li+li {{ margin-top:4px }}
    .review-tools {{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:10px; margin:14px 0 }} .review-progress {{ font-size:13px; color:var(--muted) }} .button {{ border:0; border-radius:9px; padding:10px 14px; background:var(--blue); color:#fff; font:inherit; font-size:13px; font-weight:750; cursor:pointer }} .button:hover {{ background:#1d4ed8 }}
    .review-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px }} .review-card {{ border:1px solid var(--line); border-radius:12px; padding:14px; min-width:0 }} .review-card.done {{ border-color:#86efac; background:#f0fdf4 }} .review-card.uncertain {{ border-color:#fcd34d; background:#fffbeb }} .review-card-head {{ display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px }} .priority {{ color:#64748b; font-size:12px; white-space:nowrap }} .review-visual {{ position:relative; width:min(100%,420px); margin:0 auto 10px; overflow:hidden; border-radius:8px; background:#e9eef5 }} .review-visual img {{ display:block; width:100%; height:auto }} .review-visual canvas {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none }} .review-issues {{ min-height:50px; padding-left:19px; font-size:13px }} .review-issues li+li {{ margin-top:7px }} .review-hint {{ display:block; color:var(--muted); font-size:12px; margin-top:2px }} .review-inputs {{ display:grid; grid-template-columns:1fr; gap:8px; margin-top:12px }} .review-inputs select,.review-inputs textarea {{ width:100%; max-width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; padding:9px 10px; color:var(--ink); font:inherit; font-size:13px }} .review-inputs textarea {{ min-height:62px; resize:vertical }} .review-empty {{ border:1px solid #bbf7d0; background:#f0fdf4; border-radius:12px; padding:20px; color:#166534 }} .save-status {{ font-size:12px; color:var(--muted); }}
    .issue-table {{ min-width:760px }} .issue-table th:first-child {{ width:275px }} .issue-table td:first-child {{ width:275px }} .issue-preview {{ position:relative; width:260px; max-width:100%; margin-bottom:5px; overflow:hidden; border-radius:8px; background:linear-gradient(45deg,#e9eef5 25%,transparent 25%),linear-gradient(-45deg,#e9eef5 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#e9eef5 75%),linear-gradient(-45deg,transparent 75%,#e9eef5 75%); background-size:16px 16px; background-position:0 0,0 8px,8px -8px,-8px 0; }} .issue-preview img {{ display:block; width:100%; height:auto; }} .issue-preview canvas {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none }} .image-name {{ font-size:12px; color:var(--muted); font-weight:700 }} .no-image {{ width:260px; height:130px; display:grid; place-items:center; background:#f1f5f9; color:var(--muted); border-radius:8px; font-size:12px }} .detail-extra {{ margin-top:5px; color:var(--muted); font:12px ui-monospace,monospace; word-break:break-word }} .problem-line {{ display:inline-block; margin-top:4px; color:#92400e; background:#fef3c7; border-radius:5px; padding:2px 6px; font-size:11px; font-weight:700 }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:10px }} table {{ border-collapse:collapse; width:100%; font-size:13px }} th {{ text-align:left; color:#526174; background:#f8fafc; font-size:11px; letter-spacing:.05em; text-transform:uppercase; white-space:nowrap }} td,th {{ padding:10px 11px; border-bottom:1px solid #edf1f5; vertical-align:top }} tbody tr:last-child td {{ border-bottom:0 }} tbody tr:hover {{ background:#f8fbff }} .num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap }}
    td code {{ color:#344256; background:#f1f5f9; border-radius:5px; padding:2px 5px; white-space:nowrap }} .badge {{ display:inline-block; border-radius:999px; padding:2px 7px; font-size:11px; font-weight:800 }} .badge.error {{ color:var(--red); background:#fee2e2 }} .badge.warn {{ color:var(--amber); background:#fef3c7 }} .badge.info {{ color:var(--blue); background:#dbeafe }}
    .empty {{ color:var(--muted); padding:14px; }} .files {{ display:flex; flex-wrap:wrap; gap:8px }} .file-link {{ display:flex; gap:10px; align-items:center; justify-content:space-between; border:1px solid var(--line); border-radius:9px; padding:9px 11px; color:#334155; text-decoration:none; font-size:13px }} .file-link:hover {{ border-color:#93b4f8; color:var(--blue); background:#f8fbff }} .file-link span {{ color:var(--muted) }}
    .footer {{ text-align:center; font-size:12px; color:var(--muted); padding:12px }}
    @media(max-width:850px) {{ .grid {{ grid-template-columns:repeat(2,1fr) }} .hero {{ grid-template-columns:1fr }} .export-box {{ border-left:0; border-top:1px solid var(--line); padding:15px 0 0 }} .two-col {{ grid-template-columns:1fr }} }}
    @media(max-width:700px) {{ .review-grid {{ grid-template-columns:1fr }} }}
    @media(max-width:580px) {{ .wrap {{ width:calc(100% - 22px); margin-top:20px }} header {{ display:block }} .stamp {{ text-align:left; margin-top:12px }} .grid,.phases {{ grid-template-columns:1fr }} .panel {{ padding:15px }} }}
  </style>
</head>
<body>
<main class="wrap">
  <header><div><div class="eyebrow">AutoQC · Báo cáo lần chạy gần nhất</div><h1>Kiểm tra nhanh pipeline</h1><p class="muted">Tổng hợp kết luận, cảnh báo, sửa tự động và hàng đợi người duyệt.</p></div><div class="stamp">Tạo lúc<br><strong>{report_data["generated"]}</strong></div></header>
  <section class="hero {decision_class}">
    <div><span class="decision {decision_class}">● {escape(decision)}</span><h2>Kết luận kiểm soát chất lượng</h2><ul class="reasons">{reason_html}</ul></div>
    <div class="export-box"><div class="eyebrow">Dataset</div><strong>{escape(export_text)}</strong><p class="muted">{escape(export_detail)}</p></div>
  </section>
  <section class="grid">
    <article class="card"><div class="card-label">Ảnh ở vòng cuối</div><div class="card-value">{total_images}</div><div class="card-foot">ảnh được kiểm tra sau duyệt</div></article>
    <article class="card"><div class="card-label">Lỗi bắt buộc</div><div class="card-value">{n_errors}</div><div class="card-foot">trên danh sách QC cuối</div></article>
    <article class="card"><div class="card-label">Ảnh có cảnh báo</div><div class="card-value">{warn_images}</div><div class="card-foot">chiếm {ratio_pct} số ảnh vòng cuối</div></article>
    <article class="card"><div class="card-label">Đã duyệt</div><div class="card-value">{reviewed}<span style="font-size:16px;color:#64748b"> / {budget}</span></div><div class="progress"><i style="width:{min(100, round((reviewed / budget * 100) if budget else 0))}%"></i></div><div class="card-foot">ngân sách người duyệt</div></article>
  </section>
  <section class="panel" id="manual-review"><div class="panel-head"><div><h2>Bạn cần kiểm tra gì?</h2><p class="muted">{len(review_items)} ảnh còn cảnh báo chưa được ghi nhận đã duyệt; đây là phần đang khiến kết quả Hold.</p></div></div>
    <div class="review-intro"><strong>Cách kiểm tra một ảnh</strong><ol class="review-steps"><li>Đối chiếu hình với box xanh (nhãn) và box cam (dự đoán AI); viền vàng đánh dấu dòng nhãn có vấn đề.</li><li>Đọc từng mã cảnh báo bên dưới, rồi chọn kết quả: cần sửa nhãn, cảnh báo sai, hoặc chưa chắc.</li><li>Ghi chú nếu cần. Kết quả tự lưu trong trình duyệt; tải CSV để giữ bản sao và xử lý các nhãn cần sửa.</li></ol><p class="muted" style="margin-top:9px;font-size:12px">Bảng kiểm ghi nhận nhận xét của bạn; nó không tự sửa nhãn hay đổi kết luận pipeline.</p></div>
    <div class="review-tools"><div><span class="review-progress" id="reviewProgress"></span><br><span class="save-status" id="reviewSaveStatus">Lựa chọn được lưu tạm trong trình duyệt.</span></div><button class="button" id="downloadReview">Tải kết quả kiểm tra CSV</button></div>
    <div class="review-grid" id="reviewCards"></div>
  </section>
  <section class="panel"><div class="panel-head"><div><h2>Thay đổi qua các vòng QC</h2><p class="muted">Số dòng lỗi/cảnh báo và ảnh có ghi nhận ở mỗi vòng.</p></div><span class="muted">Vào nhận dữ liệu: {intake_count} dòng · Sửa tự động: {fix_count} dòng</span></div><div class="phases">{phase_cards}</div></section>
  <div class="two-col">
    <section class="panel"><div class="panel-head"><div><h2>Danh sách vòng cuối</h2><p class="muted">{escape(final_issue_note)} Nhấp ảnh để mở ảnh gốc.</p></div><a class="file-link" href="07_error_list_v3.csv">Mở CSV ↗</a></div>
      <div class="visual-toggles"><label><input id="showLabels" type="checkbox" checked> Hiện box nhãn</label><label><input id="showPredictions" type="checkbox" checked> Hiện box AI</label><span>Viền vàng chỉ dòng nhãn đang được báo lỗi.</span></div>
      <div class="toolbar"><input id="filter" type="search" placeholder="Tìm trong mô tả…" aria-label="Tìm lỗi"><select id="filterSeverity" aria-label="Lọc mức"><option value="">Tất cả mức</option><option value="error">Lỗi</option><option value="warn">Cảnh báo</option></select><select id="filterCode" aria-label="Lọc mã lỗi">{code_options}</select><select id="filterCheck" aria-label="Lọc nhóm kiểm tra">{check_options}</select><select id="filterImage" aria-label="Lọc ảnh">{image_options}</select><span id="count" class="muted"></span></div>
      <div class="table-wrap"><table class="issue-table"><thead><tr><th>Ảnh và bounding box</th><th>Mức</th><th>Mã lỗi</th><th>Chi tiết</th></tr></thead><tbody id="issues"></tbody></table></div>
    </section>
    <section class="panel"><div class="panel-head"><div><h2>Hàng đợi người duyệt</h2><p class="muted">{len(queue)} ảnh đang chờ; hiển thị tối đa 10 ảnh đầu.</p></div><a class="file-link" href="05_review_queue.csv">Mở CSV ↗</a></div>
      <div class="table-wrap"><table><thead><tr><th>Ảnh</th><th class="num">Ưu tiên</th><th>Mã</th></tr></thead><tbody>{queue_preview}</tbody></table></div>
      <h3 style="margin-top:22px">Số box theo lớp</h3><div class="table-wrap"><table><thead><tr><th>Mã lớp</th><th class="num">Số box</th></tr></thead><tbody>{class_rows}</tbody></table></div>
    </section>
  </div>
  <section class="panel"><div class="panel-head"><div><h2>Các file báo cáo gốc</h2><p class="muted">Mở CSV hoặc TXT để xem toàn bộ dữ liệu của từng bước.</p></div></div><div class="files">{links}</div></section>
  <div class="footer">Tạo tự động từ submission/reports sau lệnh <code>make run</code>.</div>
</main>
<script>
  const report = {embedded};
  const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
  const body = document.getElementById("issues");
  const filter = document.getElementById("filter");
  const showLabels = document.getElementById("showLabels");
  const showPredictions = document.getElementById("showPredictions");
  const filterSeverity = document.getElementById("filterSeverity");
  const filterCode = document.getElementById("filterCode");
  const filterCheck = document.getElementById("filterCheck");
  const filterImage = document.getElementById("filterImage");
  const reviewCards = document.getElementById("reviewCards");
  const reviewProgress = document.getElementById("reviewProgress");
  const reviewSaveStatus = document.getElementById("reviewSaveStatus");
  const reviewStorageKey = `autoqc-review-${{report.runKey}}`;
  const reviewHelp = {{
    EMPTY_LABEL: "Ảnh nền trống hay bị thiếu toàn bộ nhãn?",
    POSSIBLE_MISSING: "AI nghi thấy vật chưa có box nhãn; nếu đúng cần thêm box.",
    POSSIBLE_LOOSE_BOX: "Box nhãn có ôm sát đúng vật không? Cân nhắc chỉnh tọa độ.",
    POSSIBLE_WRONG_CLASS: "Đối chiếu vật với lớp đã gán; sửa class nếu nhầm.",
    POSSIBLE_FALSE_POS: "Box nhãn có trỏ tới vật thật không? Nếu không, cân nhắc bỏ box.",
    BOX_TINY: "Box rất nhỏ; kiểm tra box có đang bao đúng vật không.",
    BOX_HUGE: "Box chiếm gần toàn ảnh; xác nhận vật có thực sự lớn như vậy không."
  }};
  let reviewState = {{}};
  try {{ reviewState = JSON.parse(localStorage.getItem(reviewStorageKey) || "{{}}") || {{}}; }} catch {{ reviewState = {{}}; }}
  function drawOverlay(img) {{
    const meta = report.images[img.dataset.stem];
    const canvas = img.parentElement.querySelector("canvas");
    if (!meta || !canvas || !img.naturalWidth) return;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    const lineNos = new Set((img.dataset.lines || "").split(",").map(Number).filter(Boolean));
    const draw = (box, color, tag, dashed, selected = false) => {{
      const x = (box.cx - box.w / 2) * canvas.width;
      const y = (box.cy - box.h / 2) * canvas.height;
      const w = box.w * canvas.width;
      const h = box.h * canvas.height;
      ctx.save();
      ctx.strokeStyle = selected ? "#ffd43b" : color;
      ctx.lineWidth = Math.max(2, canvas.width * (selected ? 0.006 : 0.004));
      ctx.setLineDash(dashed ? [Math.max(5, canvas.width * 0.012), Math.max(3, canvas.width * 0.006)] : []);
      ctx.strokeRect(x, y, w, h);
      const label = `${{tag}} ${{box.class_name}}${{box.conf == null ? "" : " " + box.conf.toFixed(2)}}`;
      ctx.font = `bold ${{Math.max(12, canvas.width * 0.023)}}px system-ui, sans-serif`;
      const textWidth = ctx.measureText(label).width;
      const labelY = Math.max(0, y - Math.max(20, canvas.width * 0.035));
      ctx.fillStyle = selected ? "#ffd43b" : color;
      ctx.fillRect(Math.max(0, x), labelY, textWidth + 9, Math.max(19, canvas.width * 0.035));
      ctx.fillStyle = "#111827";
      ctx.fillText(label, Math.max(4, x + 4), labelY + Math.max(14, canvas.width * 0.026));
      ctx.restore();
    }};
    if (showLabels.checked) for (const box of meta.labels) draw(box, "#22c55e", "GT", false, lineNos.has(box.line));
    if (showPredictions.checked) for (const box of meta.predictions) draw(box, "#ff8b24", "AI", true);
  }}
  function renderIssues() {{
    const query = filter.value.trim().toLocaleLowerCase();
    const rows = report.issues.filter(row =>
      [row.image,row.code,row.message,row.check,row.severity].join(" ").toLocaleLowerCase().includes(query) &&
      (!filterSeverity.value || row.severity === filterSeverity.value) &&
      (!filterCode.value || row.code === filterCode.value) &&
      (!filterCheck.value || row.check === filterCheck.value) &&
      (!filterImage.value || row.image === filterImage.value)
    );
    document.getElementById("count").textContent = `${{rows.length}} / ${{report.issues.length}} dòng`;
    body.innerHTML = rows.length ? rows.map(row => {{
      const meta = report.images[row.image] || {{}};
      const imageCell = meta.src ? `<a href="${{escapeHtml(meta.src)}}" target="_blank" rel="noopener" title="Mở ảnh gốc"><div class="issue-preview"><img src="${{escapeHtml(meta.src)}}" alt="${{escapeHtml(row.image)}}" loading="lazy" data-stem="${{escapeHtml(row.image)}}" data-lines="${{escapeHtml(row.line)}}" onload="drawOverlay(this)"><canvas aria-hidden="true"></canvas></div></a><div class="image-name">${{escapeHtml(row.image)}}${{row.line ? ` · dòng nhãn ${{escapeHtml(row.line)}}` : ""}}</div>` : `<div class="no-image">Không tìm thấy ảnh cho ${{escapeHtml(row.image)}}</div>`;
      let extra = "";
      try {{ if (row.extra) extra = JSON.stringify(JSON.parse(row.extra)); }} catch {{ extra = row.extra || ""; }}
      return `<tr><td>${{imageCell}}</td><td><span class="badge ${{row.severity === "error" ? "error" : row.severity === "warn" ? "warn" : "info"}}">${{escapeHtml(row.severity)}}</span></td><td><code>${{escapeHtml(row.code)}}</code></td><td>${{escapeHtml(row.message)}}<div class="muted">${{escapeHtml(row.check)}}</div>${{row.line ? `<span class="problem-line">Dòng nhãn ${{escapeHtml(row.line)}} được tô vàng</span>` : ""}}${{extra ? `<div class="detail-extra">Thông tin: ${{escapeHtml(extra)}}</div>` : ""}}</td></tr>`;
    }}).join("") : '<tr><td colspan="4" class="empty">Không có dòng nào khớp bộ lọc.</td></tr>';
    for (const img of body.querySelectorAll("img")) if (img.complete) drawOverlay(img);
  }}
  filter.addEventListener("input", renderIssues);
  for (const select of [filterSeverity, filterCode, filterCheck, filterImage]) select.addEventListener("change", renderIssues);
  showLabels.addEventListener("change", () => {{ renderIssues(); renderReviewCards(); }});
  showPredictions.addEventListener("change", () => {{ renderIssues(); renderReviewCards(); }});

  function reviewValue(image, field) {{ return reviewState[image]?.[field] || ""; }}
  function persistReviewState() {{
    try {{ localStorage.setItem(reviewStorageKey, JSON.stringify(reviewState)); reviewSaveStatus.textContent = "Đã lưu tạm trong trình duyệt."; }}
    catch {{ reviewSaveStatus.textContent = "Trình duyệt không cho lưu tự động; hãy tải CSV để giữ kết quả."; }}
  }}
  function saveReview(image, field, value) {{
    reviewState[image] ||= {{}};
    reviewState[image][field] = value;
    persistReviewState();
    renderReviewCards();
  }}
  function renderReviewCards() {{
    const items = report.reviewItems;
    const recorded = items.filter(item => reviewValue(item.image, "decision")).length;
    const uncertain = items.filter(item => reviewValue(item.image, "decision") === "uncertain").length;
    reviewProgress.textContent = `${{recorded}} / ${{items.length}} ảnh đã ghi nhận · ${{uncertain}} ảnh chưa chắc`;
    if (!items.length) {{
      reviewCards.innerHTML = '<div class="review-empty">Không còn cảnh báo chưa được ghi nhận đã duyệt trong báo cáo này.</div>';
      return;
    }}
    reviewCards.innerHTML = items.map(item => {{
      const meta = report.images[item.image] || {{}};
      const lineNos = item.issues.map(issue => issue.line).filter(Boolean).join(",");
      const issueList = item.issues.map(issue => `<li><code>${{escapeHtml(issue.code)}}</code> — ${{escapeHtml(issue.message)}}<span class="review-hint">${{escapeHtml(reviewHelp[issue.code] || "Xem ảnh và xác nhận cảnh báo có đúng với nhãn không.")}}</span></li>`).join("");
      const image = meta.src ? `<a href="${{escapeHtml(meta.src)}}" target="_blank" rel="noopener" title="Mở ảnh gốc"><div class="review-visual"><img src="${{escapeHtml(meta.src)}}" alt="${{escapeHtml(item.image)}}" loading="lazy" data-stem="${{escapeHtml(item.image)}}" data-lines="${{escapeHtml(lineNos)}}" onload="drawOverlay(this)"><canvas aria-hidden="true"></canvas></div></a>` : `<div class="no-image">Không tìm thấy ảnh</div>`;
      const decision = reviewValue(item.image, "decision");
      const note = reviewValue(item.image, "note");
      const cardStatus = decision === "uncertain" ? "uncertain" : decision ? "done" : "";
      return `<article class="review-card ${{cardStatus}}" data-image="${{escapeHtml(item.image)}}"><div class="review-card-head"><strong>${{escapeHtml(item.image)}}</strong><span class="priority">Ưu tiên ${{Number(item.priority).toFixed(2)}}</span></div>${{image}}<ul class="review-issues">${{issueList}}</ul><div class="review-inputs"><select class="review-decision" aria-label="Kết quả kiểm tra ${{escapeHtml(item.image)}}" data-image="${{escapeHtml(item.image)}}"><option value="">Chọn kết quả kiểm tra…</option><option value="fix" ${{decision === "fix" ? "selected" : ""}}>Cảnh báo đúng — cần sửa nhãn</option><option value="false_alarm" ${{decision === "false_alarm" ? "selected" : ""}}>Cảnh báo sai — nhãn ổn</option><option value="uncertain" ${{decision === "uncertain" ? "selected" : ""}}>Chưa chắc — cần xem lại</option></select><textarea class="review-note" data-image="${{escapeHtml(item.image)}}" placeholder="Ghi chú (không bắt buộc)…">${{escapeHtml(note)}}</textarea></div></article>`;
    }}).join("");
    for (const img of reviewCards.querySelectorAll("img")) if (img.complete) drawOverlay(img);
    reviewCards.querySelectorAll(".review-decision").forEach(select => select.addEventListener("change", () => saveReview(select.dataset.image, "decision", select.value)));
    reviewCards.querySelectorAll(".review-note").forEach(textarea => textarea.addEventListener("input", () => {{
      const image = textarea.dataset.image;
      reviewState[image] ||= {{}};
      reviewState[image].note = textarea.value;
      persistReviewState();
    }}));
  }}
  function downloadReviewCsv() {{
    const quoteCsv = value => `"${{String(value ?? "").replaceAll('"', '""')}}"`;
    const lines = [["image", "priority", "decision", "notes", "codes", "issues"].map(quoteCsv).join(",")];
    for (const item of report.reviewItems) {{
      lines.push([item.image, item.priority, reviewValue(item.image, "decision"), reviewValue(item.image, "note"), item.issues.map(issue => issue.code).join(";"), item.issues.map(issue => issue.message).join(" | ")].map(quoteCsv).join(","));
    }}
    const blob = new Blob(["\\ufeff" + lines.join("\\r\\n")], {{type:"text/csv;charset=utf-8"}});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "kiem-tra-thu-cong.csv";
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }}
  document.getElementById("downloadReview").addEventListener("click", downloadReviewCsv);
  renderReviewCards();
  renderIssues();
</script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default="submission/reports", help="thư mục báo cáo pipeline")
    parser.add_argument("--output", help="đường dẫn HTML đầu ra (mặc định: <reports>/report.html)")
    args = parser.parse_args()
    reports_dir = Path(args.reports)
    if not reports_dir.is_absolute():
        reports_dir = ROOT / reports_dir
    output = Path(args.output) if args.output else reports_dir / "report.html"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(reports_dir), encoding="utf-8")
    print(f"Báo cáo HTML: {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")


if __name__ == "__main__":
    main()
