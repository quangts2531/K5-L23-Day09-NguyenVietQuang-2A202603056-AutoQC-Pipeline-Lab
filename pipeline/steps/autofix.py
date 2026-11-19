"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401
from pathlib import Path

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    """Apply safe fixes for ERROR issues and return an audit row per change.

    Line numbers in issues refer to the original label file, so fixes are
    collected by file and each changed file is written once.
    """
    root = Path(ds.root).resolve()
    rows = []

    def value(issue, key, default=None):
        if isinstance(issue, dict):
            return issue.get(key, default)
        return getattr(issue, key, default)

    def is_error(issue):
        return str(value(issue, "severity", "")).lower() == ERROR

    def safe_stem(stem):
        # Issue names are normally stems produced by Dataset. Guard against a
        # malformed issue being used to address a file outside the dataset.
        return (
            isinstance(stem, str)
            and stem not in ("", ".", "..")
            and Path(stem).name == stem
            and "/" not in stem
            and "\\" not in stem
        )

    def inside_dataset(path):
        try:
            return Path(path).resolve().is_relative_to(root)
        except (OSError, ValueError):
            return False

    def image_paths(stem):
        if not safe_stem(stem):
            return []
        return [
            path for path in ds.image_files()
            if path.stem == stem and inside_dataset(path)
        ]

    def label_path(stem):
        if not safe_stem(stem):
            return None
        path = ds.label_path(stem)
        return path if inside_dataset(path) else None

    def fmt(number):
        # Avoid long binary floating point tails in the human-readable audit
        # and keep enough precision for normalized box coordinates.
        return format(number, ".12g")

    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    remove_stems = set()
    for issue in issues:
        if not is_error(issue):
            continue
        code = value(issue, "code", "")
        stem = value(issue, "image", "")
        if code.startswith("DUP_IMAGE_") or code in remove_codes:
            if safe_stem(stem):
                remove_stems.add(stem)

    # Whole-image removal takes precedence over any label-level issue for the
    # same stem. Remove all same-stem image files, including alternate formats.
    for stem in sorted(remove_stems):
        images = image_paths(stem)
        label = label_path(stem)
        existing_images = [p for p in images if p.exists()]
        existing_label = label is not None and label.exists()
        if not existing_images and not existing_label:
            continue
        before = "; ".join(
            [p.name for p in existing_images]
            + ([label.name] if existing_label else [])
        )
        for path in existing_images:
            path.unlink(missing_ok=True)
        if existing_label:
            label.unlink(missing_ok=True)
        rows.append({
            "image": stem,
            "line": None,
            "action": "remove_image",
            "before": before,
            "after": "đã xóa khỏi ds",
        })

    # Collect original line numbers and codes before changing any label file.
    line_issues = defaultdict(lambda: defaultdict(set))
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    for issue in issues:
        if not is_error(issue):
            continue
        stem = value(issue, "image", "")
        code = value(issue, "code", "")
        line_no = value(issue, "line")
        if stem in remove_stems or not safe_stem(stem):
            continue
        if code in drop_codes or code == "BOX_OUT_OF_IMAGE":
            if isinstance(line_no, int) and line_no > 0:
                line_issues[stem][line_no].add(code)

    for stem in sorted(line_issues):
        path = label_path(stem)
        if path is None or not path.is_file():
            continue
        original = path.read_text(encoding="utf-8", errors="replace")
        original_lines = original.splitlines()
        changed = False

        # Enumerate the original lines, including blank lines, to preserve the
        # line-number coordinate system used by the issues.
        for line_no, codes in sorted(line_issues[stem].items()):
            index = line_no - 1
            if index >= len(original_lines):
                continue
            before = original_lines[index]
            if not before.strip():
                continue

            # A drop always wins over clipping when a single line has multiple
            # errors. Invalid/missing rows are ignored rather than guessed at.
            if codes & drop_codes:
                original_lines[index] = None
                rows.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": before,
                    "after": "",
                })
                changed = True
                continue

            if "BOX_OUT_OF_IMAGE" not in codes:
                continue
            try:
                parsed = parse_yolo_line(before)
            except (TypeError, ValueError):
                continue
            class_id, cx, cy, width, height = parsed
            left = max(0.0, cx - width / 2)
            right = min(1.0, cx + width / 2)
            top = max(0.0, cy - height / 2)
            bottom = min(1.0, cy + height / 2)

            # A box fully outside the image has no remaining area after clip.
            if right <= left or bottom <= top:
                original_lines[index] = None
                rows.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": before,
                    "after": "",
                })
            else:
                clipped = " ".join((
                    before.split()[0],
                    fmt((left + right) / 2),
                    fmt((top + bottom) / 2),
                    fmt(right - left),
                    fmt(bottom - top),
                ))
                original_lines[index] = clipped
                rows.append({
                    "image": stem,
                    "line": line_no,
                    "action": "clip_box",
                    "before": before,
                    "after": clipped,
                })
            changed = True

        if changed:
            had_final_newline = original.endswith(("\n", "\r"))
            content = "\n".join(line for line in original_lines if line is not None)
            if had_final_newline and content:
                content += "\n"
            path.write_text(content, encoding="utf-8")

    # Missing labels are intentionally left missing so an empty file cannot be
    # mistaken for a reviewed background image.
    missing_stems = {
        value(issue, "image", "")
        for issue in issues
        if is_error(issue)
        and value(issue, "code", "") == "MISSING_LABEL_FILE"
        and value(issue, "image", "") not in remove_stems
    }
    for stem in sorted(missing_stems):
        if not safe_stem(stem) or not image_paths(stem):
            continue
        label = label_path(stem)
        if label is None or label.exists():
            continue
        rows.append({
            "image": stem,
            "line": None,
            "action": "needs_label",
            "before": "(không có file)",
            "after": "chuyển người gán",
        })

    # A BAD_CLASS_NAME may be reported once per mismatch. Repair the names map
    # once and preserve all other keys in data.yaml.
    if any(
        is_error(issue) and value(issue, "code", "") == "BAD_CLASS_NAME"
        for issue in issues
    ):
        yaml_path = Path(ds.yaml_path)
        if inside_dataset(yaml_path):
            try:
                document = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path.exists() else {}
            except (OSError, yaml.YAMLError):
                document = {}
            if not isinstance(document, dict):
                document = {}

            classes = list(cfg["classes"])
            old_names = document.get("names")
            wanted = {i: name for i, name in enumerate(classes)}
            current = None
            if isinstance(old_names, list):
                current = {i: name for i, name in enumerate(old_names)}
            elif isinstance(old_names, dict):
                try:
                    current = {int(i): name for i, name in old_names.items()}
                except (TypeError, ValueError):
                    current = None

            if current != wanted:
                before = yaml.safe_dump(old_names, allow_unicode=True, sort_keys=False).strip() if old_names is not None else "(không có names)"
                document["names"] = wanted
                yaml_path.write_text(
                    yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                after = yaml.safe_dump(wanted, allow_unicode=True, sort_keys=False).strip()
                rows.append({
                    "image": "data.yaml",
                    "line": None,
                    "action": "rename_classes",
                    "before": before,
                    "after": after,
                })

    rows.sort(key=lambda row: (
        row["image"],
        -1 if row["line"] is None else row["line"],
        row["action"],
    ))
    return rows
