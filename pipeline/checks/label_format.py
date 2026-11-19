"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def check_box_tiny(ds, stem, line_no, w, h, cfg):
    """Trả về cảnh báo nếu cạnh ngắn của box nhỏ hơn ngưỡng pixel."""
    image_size = ds.image_size(stem)
    if image_size is None:
        return None

    image_width, image_height = image_size
    short_side_px = min(w * image_width, h * image_height)
    min_box_px = cfg["label_format"]["min_box_px"]
    if short_side_px >= min_box_px:
        return None

    return Issue(
        check="label_format",
        severity=WARN,
        code="BOX_TINY",
        image=stem,
        line=line_no,
        message=f"cạnh ngắn {short_side_px:.2f}px < {min_box_px}px",
    )


def check_box_huge(stem, line_no, w, h, cfg):
    """Trả về cảnh báo nếu box chiếm diện tích lớn hơn ngưỡng cho phép."""
    box_area_frac = w * h
    max_box_frac = cfg["label_format"]["max_box_frac"]
    if box_area_frac <= max_box_frac:
        return None

    return Issue(
        check="label_format",
        severity=WARN,
        code="BOX_HUGE",
        image=stem,
        line=line_no,
        message=f"diện tích box {box_area_frac:.4f} > {max_box_frac}",
    )


def run(ds, cfg):
    # TODO: duyệt từng file nhãn, từng dòng
    class_index = [i for i in range(len(cfg["classes"])) ]
    issues = []
    for label_file in ds.label_files():
        stem = label_file.stem
        list_box = []
        # # Mỗi item là (số_dòng, nội_dung_dòng)
        for line_no, text in ds.read_label_lines(stem) or []:
            try:
                cls, cx, cy, w, h = parse_yolo_line(text)
                if cls not in class_index:
                    issues.append(Issue(
                        check = "label_format",
                        severity=ERROR,
                        code = "BAD_CLASS_ID",
                        image = stem,
                        line=line_no,
                        message="Id label ko hợp lệ",
                    ))
                    continue
                if  w <=0 or h<=0:
                    issues.append(Issue(
                        check="label_format",
                        severity=ERROR,
                        code="BOX_ZERO",
                        image=stem,
                        line=line_no,
                        message="Chiều cao chưa hợp lệ ",
                    ))
                    continue
                left, top = cx - w / 2, cy - h / 2
                right, bottom = cx + w / 2, cy + h / 2
                overflow = max(0.0, -left, -top, right - 1.0, bottom - 1.0)
                if overflow > cfg["label_format"]["overflow_tol"]:
                    issues.append(Issue(
                        check="label_format",
                        severity=ERROR,
                        code="BOX_OUT_OF_IMAGE",
                        image=stem,
                        line=line_no,
                        extra={"overflow": overflow},
                        message=f'mép box vượt [0,1] quá `{cfg["label_format"]["overflow_tol"]}`',
                    ))
                    continue

                tiny_issue = check_box_tiny(ds, stem, line_no, w, h, cfg)
                if tiny_issue is not None:
                    issues.append(tiny_issue)

                huge_issue = check_box_huge(stem, line_no, w, h, cfg)
                if huge_issue is not None:
                    issues.append(huge_issue)

                dup_box = [box[0] == cls and iou([box[1], box[2], box[3], box[4]], [cx, cy, w, h])>= cfg["label_format"]["dup_box_iou"] for box in list_box]

                if any(dup_box):
                    issues.append(Issue(
                        check="label_format",
                        severity=ERROR,
                        code="DUP_BOX",
                        image=stem,
                        line=line_no,
                        message=f'cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `{cfg["label_format"]["dup_box_iou"]}`',
                    ))
                    continue

                list_box.append([cls, cx, cy, w, h])
            except ValueError as e:
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BAD_FORMAT",
                    image=stem,
                    line=line_no,
                    message="không đúng 5 cột số ",
                ))
                continue
    return issues

def iou(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2

    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih

    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
