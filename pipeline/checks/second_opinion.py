"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
import math

from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    thresholds = cfg["second_opinion"]
    min_conf = thresholds["conf"]
    min_area_px = thresholds["min_area_px"]
    fp_conf = thresholds["fp_conf"]
    iou_missing = thresholds["iou_missing"]
    iou_match = thresholds["iou_match"]
    iou_loose = thresholds["iou_loose"]
    issues = []

    def normalized_predictions(rows):
        """Return valid prediction rows as (class, conf, cx, cy, w, h)."""
        result = []
        for row in rows or []:
            try:
                raw_cls, raw_conf, cx, cy, width, height = row
                raw_class_id = float(raw_cls)
                conf, cx, cy, width, height = map(
                    float, (raw_conf, cx, cy, width, height)
                )
            except (TypeError, ValueError, OverflowError):
                continue
            values = (conf, cx, cy, width, height)
            if (
                not math.isfinite(raw_class_id)
                or not raw_class_id.is_integer()
                or not all(math.isfinite(v) for v in values)
            ):
                continue
            class_id = int(raw_class_id)
            if width <= 0 or height <= 0:
                continue
            result.append((class_id, conf, cx, cy, width, height))
        return result

    for image_path in ds.image_files():
        stem = image_path.stem
        if stem not in preds:
            continue
        label_path = ds.label_path(stem)
        if not label_path.is_file():
            continue

        # Keep original line numbers while excluding malformed rows and class
        # IDs that Dataset.valid_boxes would reject.
        label_rows = []
        for line_no, text in ds.read_label_lines(stem) or []:
            try:
                box = parse_yolo_line(text)
            except (TypeError, ValueError):
                continue
            if 0 <= box[0] < len(ds.classes):
                label_rows.append((line_no, box))
        labels = [box for _, box in label_rows]
        all_predictions = normalized_predictions(preds[stem])

        # Main review flags use the stricter confidence and minimum pixel area.
        image_size = ds.image_size(stem)
        if image_size is None:
            width_px = height_px = None
        else:
            width_px, height_px = image_size
        main_predictions = []
        if width_px is not None:
            main_predictions = [
                prediction
                for prediction in all_predictions
                if prediction[1] >= min_conf
                and prediction[4] * width_px * prediction[5] * height_px >= min_area_px
            ]

        for class_id, conf, cx, cy, width, height in main_predictions:
            predicted_box = (cx, cy, width, height)
            candidates = []
            for label in labels:
                overlap = iou_xywh(predicted_box, label[1:])
                if overlap >= iou_missing:
                    candidates.append((label, overlap))

            if not candidates:
                issues.append(Issue(
                    "second_opinion", WARN, "POSSIBLE_MISSING", stem,
                    f"model thấy lớp {class_id} (conf={conf:.3f}) nhưng không có box nhãn gần",
                    extra={"conf": conf},
                ))
                continue

            same_class = [iou for label, iou in candidates if label[0] == class_id]
            if same_class:
                best_iou = max(same_class)
                if best_iou < iou_loose:
                    issues.append(Issue(
                        "second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem,
                        f"box nhãn cùng lớp khớp lỏng (IoU={best_iou:.3f})",
                        extra={"conf": conf, "iou": best_iou},
                    ))
                # A same-class candidate resolves this prediction even when a
                # different class overlaps it more strongly.
                continue

            other_class = [iou for label, iou in candidates if label[0] != class_id]
            if other_class and max(other_class) >= iou_match:
                best_iou = max(other_class)
                issues.append(Issue(
                    "second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem,
                    f"box khớp với lớp nhãn khác (IoU={best_iou:.3f})",
                    extra={"conf": conf, "iou": best_iou},
                ))

        # False positives use every prediction above fp_conf, including small
        # boxes that were excluded from the main missing/loose/wrong-class pass.
        fp_predictions = [p for p in all_predictions if p[1] >= fp_conf]
        false_positives = []
        for line_no, label in label_rows:
            best_iou = max(
                (iou_xywh(label[1:], prediction[2:]) for prediction in fp_predictions),
                default=0.0,
            )
            if best_iou < iou_missing:
                false_positives.append(Issue(
                    "second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                    f"box nhãn dòng {line_no} không khớp dự đoán đủ tin cậy (IoU tối đa={best_iou:.3f})",
                    line=line_no,
                    extra={"iou": best_iou},
                ))
        issues.extend(false_positives)

    return issues
