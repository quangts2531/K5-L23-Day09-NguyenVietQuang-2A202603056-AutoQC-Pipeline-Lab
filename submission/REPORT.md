# Báo cáo Day 09 — Nguyễn Viết Quang, 2A202603056

## 1. Kết luận trước

Pipeline kết luận **Hold**. Trong 224 ảnh raw, bước nhận dữ liệu loại 19 ảnh, còn 205 ảnh; cả 205 ảnh được xuất vào `clean_v1` dưới dạng bản Hold. Sau lượt duyệt mô phỏng 80 ảnh, còn 0 lỗi bắt buộc nhưng 38/205 ảnh vẫn có cảnh báo (18,54%), cao hơn ngưỡng 10%, nên bộ dữ liệu chưa được phát hành.

## 2. Lỗi và phân việc

`02_error_list_v1.csv` có 208 dòng: 55 lỗi và 153 cảnh báo. Bảng ghi **số dòng / số ảnh**; `CLASS_RARE` là cảnh báo cấp dataset.

| Mã lỗi | Dòng / ảnh | Ai xử lý | Phân tích |
|---|---:|---|---|
| `BAD_CLASS_ID` | 6 / 6 | Máy | Bỏ dòng nhãn có ID lớp không hợp lệ. |
| `BAD_CLASS_NAME` | 2 / 1 | Máy | Chuẩn hóa `Person`, `Bus` thành tên lớp đúng; 1 thay đổi `data.yaml`. |
| `BAD_FORMAT` | 3 / 3 | Máy | Bỏ dòng nhãn sai định dạng. |
| `BAD_SIZE` | 3 / 3 | Máy | Loại ảnh quá nhỏ ngay ở bước nhận dữ liệu. |
| `BOX_HUGE` | 3 / 3 | Người | Xem ảnh để xác nhận box có bao gần toàn ảnh hợp lý không. |
| `BOX_OUT_OF_IMAGE` | 8 / 8 | Máy | Cắt box vào trong ảnh (`clip_box`). |
| `BOX_TINY` | 6 / 4 | Người | Kiểm tra box nhỏ có bám đúng vật không; 6 dòng nằm trên 4 ảnh. |
| `BOX_ZERO` | 4 / 4 | Máy | Bỏ dòng có chiều rộng hoặc cao bằng 0. |
| `CLASS_RARE` | 4 / 1 dataset | Máy gián tiếp | Thống kê ID lớp hiếm; các dòng ID sai được bỏ rồi thống kê lại. |
| `CORRUPT_IMAGE` | 2 / 2 | Máy | Loại ảnh không đọc được ở bước nhận dữ liệu. |
| `DUP_BOX` | 7 / 7 | Máy | Bỏ box trùng với box trước đó. |
| `DUP_IMAGE_EXACT` | 8 / 8 | Máy | Loại ảnh trùng từng byte ở bước nhận dữ liệu. |
| `DUP_IMAGE_NEAR` | 2 / 2 | Máy | Loại cặp ảnh trùng gần đúng được phát hiện. |
| `EMPTY_LABEL` | 12 / 12 | Người | Mở ảnh để phân biệt ảnh nền với ảnh bị thiếu nhãn. |
| `LEAK_TEST` | 4 / 4 | Máy | Loại ảnh trùng kho test/val để tránh rò rỉ dữ liệu. |
| `MISSING_LABEL_FILE` | 6 / 6 | Người | Máy đưa vào hàng đợi (`needs_label`), không tự tạo nhãn. |
| `POSSIBLE_FALSE_POS` | 29 / 27 | Người | So box với ảnh; mô hình có thể không nhận ra vật nhỏ hoặc khuất. |
| `POSSIBLE_LOOSE_BOX` | 28 / 24 | Người | Đối chiếu vị trí và độ khít của box với vật. |
| `POSSIBLE_MISSING` | 53 / 52 | Người | Xác nhận vật AI phát hiện có thật sự thiếu box nhãn không. |
| `POSSIBLE_WRONG_CLASS` | 18 / 18 | Người | Đối chiếu lớp nhãn với loại vật trong ảnh. |

`03_fixes.csv` ghi 35 hành động: 20 bỏ dòng, 8 cắt box, 6 chuyển ảnh thiếu nhãn cho người và 1 chuẩn hóa `data.yaml`. Hàng đợi có 125 ảnh, ngân sách duyệt 80. Trong 80 ảnh ưu tiên đầu, 74 ảnh có ít nhất một cảnh báo ngữ nghĩa `POSSIBLE_*` (88 mã cảnh báo ngữ nghĩa cộng dồn; đây chưa phải 88 lỗi đã xác nhận). `img_0020` đứng đầu với điểm 11,524 vì có `POSSIBLE_MISSING`, `POSSIBLE_WRONG_CLASS` và một dòng bị máy loại; ảnh thiếu nhãn được cộng trọng số cao (`NEEDS_LABEL` = 10).

## 3. Calibration

| Tham số | Ban đầu → đang dùng | Precision / Recall trước → sau | Lý do và đánh đổi |
|---|---|---|---|
| `duplicates.phash_dist` | 2 → **2** | Near duplicate: 1,00 / 0,50 → 1,00 / 0,50; exact duplicate: 1,00 / 1,00 → 1,00 / 1,00 | Giữ ngưỡng bảo thủ để tránh loại nhầm ảnh. Đổi lên 16 đạt recall near duplicate 0,83 trên calibration, nhưng cần xem các cặp ảnh raw trước khi chấp nhận ngưỡng rộng hơn. |
| `second_opinion.conf` | 0,25 → **0,6** | `WRONG_CLASS`: 0,56 / 0,83 → 0,62 / 0,83; `MISSING`: 0,50 / 1,00 → 0,57 / 1,00; `LOOSE_BOX`: 0,29 / 0,40 → 0,20 / 0,20; `FALSE_POS`: 0,31 / 1,00 → 0,31 / 1,00 | Ngưỡng 0,6 cải thiện precision của hai nhóm đầu và giữ recall `MISSING`; đổi lại, `LOOSE_BOX` kém hơn. `FALSE_POS` không đổi vì dùng ngưỡng `fp_conf` riêng. |

Calibration có thể đánh lừa nếu tối ưu một nhóm lỗi trên tập nhỏ: chỉ có 6 near duplicate được chèn; ở `phash_dist=16`, recall tăng lên 5/6 nhưng chưa chứng minh ngưỡng đó an toàn cho toàn bộ ảnh raw. Tương tự, tại `conf=0,6`, `LOOSE_BOX` chỉ có 1 TP, 4 FP và 4 FN (precision/recall 0,20/0,20). Vì vậy điểm calibration là gợi ý để chọn ngưỡng, không thay thế việc xem ảnh.

## 4. Cửa chặn và điểm mù

Thí nghiệm budget 0 đã có kết quả ở `work/rep_nb/08_summary.json`: **Rework**, 6 lỗi bắt buộc trên 205 ảnh, 0 ảnh được duyệt, `exported: null`. Sáu lỗi là `MISSING_LABEL_FILE`; thư mục `work/exp_nb` không được tạo. `pipeline.run` ánh xạ Rework sang mã thoát **20**, nên cửa chặn đã ngăn xuất dataset khi chưa có người gán nhãn.

Hai điểm mù có bằng chứng từ calibration:

- `POSSIBLE_LOOSE_BOX` ở ngưỡng 0,6 chỉ tìm được 1/5 lỗi đã chèn và đồng thời có 4 cảnh báo sai. Cần người xem ảnh thay vì sửa box chỉ theo gợi ý của model.
- `DUP_IMAGE_NEAR` ở `phash_dist=2` có recall 3/6 = 0,50, tức bỏ sót 3 cặp near duplicate đã chèn; một số dạng ảnh trùng khó hơn vẫn có thể lọt qua.
- `POSSIBLE_FALSE_POS` có precision 0,31 (5 TP, 11 FP ở calibration), nên cảnh báo box thừa có thể do model không phát hiện được vật thật.

## 5. Train: raw vs clean

Hai biến thể dùng cùng `yolo26n.pt`, 20 epoch, batch 16, imgsz 640, cùng thiết bị và seed 0/1; đánh giá trên cùng test. Hash manifest của cả hai kết quả `clean` khớp `submission/export/clean_v1/manifest.json`.

| Biến thể | Ảnh train | mAP50 (TB ± SD) | mAP50-95 (TB ± SD) | Precision (TB ± SD) | Recall (TB ± SD) |
|---|---:|---:|---:|---:|---:|
| raw | 224 | 0,726 ± 0,000 | 0,512 ± 0,004 | 0,764 ± 0,061 | 0,649 ± 0,028 |
| clean | 205 | 0,700 ± 0,030 | 0,486 ± 0,013 | 0,715 ± 0,073 | 0,615 ± 0,017 |

Chênh mAP50 (clean − raw) là **−0,025** (seed 0: 0,6788; seed 1: 0,7219). Mức giảm này nhỏ hơn độ lệch chuẩn giữa hai seed của clean (0,030), nên với hai seed chưa thể kết luận clean tốt hơn hay kém hơn; chỉ thấy xu hướng trung bình thấp hơn trong thí nghiệm này.

Hai nguyên nhân cần kiểm tra thêm: (1) raw còn 4 ảnh `LEAK_TEST` theo `01_intake.csv`, nên có thể làm điểm test raw cao giả tạo; chạy lại đối chứng raw sau khi loại đúng các ảnh rò rỉ. (2) clean có ít hơn raw 19 ảnh và vẫn Hold với 38/205 ảnh còn cảnh báo; AP50 lớp truck giảm từ khoảng 0,648 xuống 0,559. Đối chiếu số box theo lớp giữa raw/clean và kiểm tra nhãn truck trong các ảnh bị cảnh báo. Test chỉ có 100 ảnh và mới có hai seed, nên cần thêm seed hoặc tập test lớn hơn trước khi kết luận chắc chắn.

## 6. Quyết định và ghi chú AI

Quyết định hiện tại là **Hold**; `clean_v1` là bản xuất có trạng thái Hold, chưa phải bản phát hành. Cần kiểm tra 38 ảnh còn cảnh báo trong `reports/report.html`, sửa nhãn khi xác nhận có lỗi rồi chạy lại `make run`. Nhận xét lưu từ HTML là checklist, không tự cập nhật nhãn hay log pipeline. Để qua gate trên 205 ảnh, cần 0 lỗi bắt buộc và tối đa 20 ảnh chưa duyệt có cảnh báo (20/205 = 9,76%); tức cần xem/giải quyết ít nhất 18/38 ảnh nếu không phát sinh cảnh báo mới.

- **Dùng AI:** Có. Prompt chính: “tạo HTML để kiểm tra báo cáo sau `make run`, hiện ảnh và lỗi để đối chiếu”; tiếp theo yêu cầu thêm bộ lọc và bảng ghi nhận kiểm tra.
- **Cách kiểm:** Đối chiếu số liệu HTML với `08_summary.json` và CSV, xem lại quy tắc gate trong `pipeline/steps/gate.py`, rồi chạy script tạo HTML để cập nhật trang.
- **Một lần AI sai:** Một phép đếm tạm thời đếm các dòng lỗi lồng trong dữ liệu ảnh thành 80 thay vì 38 ảnh chưa duyệt; đã sửa bằng cách đếm ảnh duy nhất và đối chiếu `06_review_log.csv`.

**CVAT thưởng:** Chưa thực hiện; không tính điểm phần này.
