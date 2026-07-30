# 04 — Giao thức thí nghiệm

## 1. Mục tiêu

Tạo so sánh có kiểm soát giữa:

- học tập trung (centralized);
- mỗi cơ sở tự học (local-only);
- FedAvg;
- FedProx;

trên ba mức heterogeneity: IID, Dirichlet `α=0.5`, Dirichlet `α=0.1`.

## 2. Hai loại kết quả phải tách biệt

| Loại | Dữ liệu | Mục đích | Có đưa vào báo cáo kết quả? |
|---|---|---|---|
| `synthetic_smoke_only` | vector tổng hợp | kiểm tra code path/CI/API | Không |
| image experiment | PlantVillage hoặc bộ đã khai báo | trả lời RQ1–RQ4 | Có |

Mọi JSON smoke đều có trường `warning`; pipeline xuất bảng chính thức phải loại loại này.

## 3. Ma trận bắt buộc

| ID | Chế độ | Partition | Algorithm | Ưu tiên |
|---|---|---|---|---|
| CEN-MBV2 | pooled data | pooled | centralized AdamW | bắt buộc |
| LOC-MBV2 | từng client | natural partition | local-only | bắt buộc |
| FL-IID-AVG | federated | IID | FedAvg | bắt buộc |
| FL-A05-AVG | federated | α=0.5 | FedAvg | bắt buộc |
| FL-A01-AVG | federated | α=0.1 | FedAvg | bắt buộc |
| FL-A05-PROX | federated | α=0.5 | FedProx | bắt buộc |
| FL-A01-PROX | federated | α=0.1 | FedProx | bắt buộc |
| FL-A01-R18 | federated | α=0.1 | FedAvg + ResNet18 | nâng cao |

File máy đọc: `configs/experiment_matrix.csv`.

## 4. Cấu hình khởi đầu

Đây là protocol ban đầu; chỉ được chốt sau pilot trên validation:

| Tham số | Giá trị ban đầu |
|---|---|
| model | MobileNetV2, ImageNet weights |
| input | 224×224 RGB |
| optimizer local | AdamW |
| learning rate | `1e-3`; pilot thêm `3e-4` |
| weight decay | `1e-4` |
| batch size | 32; giảm nếu thiếu VRAM |
| clients | 4 |
| participation | 100% mỗi round |
| max FL rounds | 30 |
| local epochs | 1 |
| FedProx μ | pilot `{0.001, 0.01, 0.1}`, mặc định 0.01 |
| centralized epochs | 30 |
| local-only epochs | 30 |
| development seed | 2026 |
| final seeds | 2026, 2027, 2028 nếu tài nguyên cho phép |

Không chọn learning rate, μ hay số round bằng global test.

## 5. Nguyên tắc công bằng

1. Dùng cùng master split và class mapping.
2. Centralized dùng hợp của các local-train samples; không dùng local validation/test để train.
3. Các FL algorithm dùng cùng initial global checkpoint cho cùng seed.
4. Data augmentation giống nhau.
5. Evaluation transform giống nhau.
6. Cùng model architecture và optimizer trừ khi chính yếu tố đó được so sánh.
7. Báo cả total local epochs/data passes, không chỉ số round.
8. Báo hardware và simulation mode; simulation tuần tự không đại diện wall-clock deployment song song.
9. Nếu run lỗi, lưu lỗi; không âm thầm bỏ seed xấu.

## 6. Metric phân loại

Với mỗi lớp \(c\):

\[
Precision_c = \frac{TP_c}{TP_c+FP_c}
\]

\[
Recall_c = \frac{TP_c}{TP_c+FN_c}
\]

\[
F1_c = \frac{2Precision_cRecall_c}{Precision_c+Recall_c}
\]

Macro F1:

\[
MacroF1 = \frac{1}{C}\sum_{c=1}^{C}F1_c
\]

Báo:

- accuracy;
- macro precision, recall, F1;
- precision/recall/F1/support từng lớp;
- confusion matrix;
- metric toàn cục và từng client;
- worst-client macro F1.
- recall phát hiện ảnh có hại và tỷ lệ ảnh bệnh/sâu hại bị nhầm thành healthy;
- disease F1, pest F1 và F1 riêng lớp two-spotted spider mite.

Accuracy không đủ vì class imbalance có thể che hiệu năng lớp ít mẫu.

Trong Flower, metric `central_*` tính trực tiếp trên global test set cố định là nguồn
để so sánh thuật toán và dựng confusion matrix. Metric `eval_*` là trung bình có
trọng số từ validation cục bộ của client; không được diễn giải như confusion matrix
toàn cục được tái dựng từ toàn bộ dự đoán.

## 7. Metric FL/hệ thống

- train loss và validation loss theo round;
- round đạt ngưỡng tốt nhất trên validation;
- tổng số round;
- local training time;
- aggregation/evaluation time;
- total wall-clock time;
- model size;
- upload/download bytes mỗi round;
- tổng byte:

\[
Bytes_{total} = \sum_t \sum_{k \in S_t}
  (Bytes^{down}_{t,k}+Bytes^{up}_{t,k})
\]

- số client thành công/thất bại mỗi round;
- độ lệch metric giữa client.

Không dùng tốc độ của synthetic NumPy để suy ra tốc độ MobileNetV2.

## 8. Chọn model và dừng

### Pilot

- Chạy seed 2026.
- Dùng local validation/federated validation.
- Chọn LR và `μ`.
- Quan sát plateau để xác nhận 30 round/epoch có hợp lý.

### Main study

- Khóa config.
- Chạy tất cả scenario.
- Chỉ sau khi hoàn tất mới đánh giá global test.
- Nếu dùng early stopping, rule phải giống giữa các algorithm và checkpoint được chọn bằng validation.

### Báo cáo nhiều seed

Với ba seed, báo `mean ± standard deviation`. Ba seed là tối thiểu hợp lý cho đồ án nếu tài nguyên cho phép; nếu chỉ một seed, nêu đây là hạn chế và không diễn giải chênh lệch nhỏ thành kết luận mạnh.

## 9. Artifact mỗi run

```text
artifacts/runs/<run-id>/
├── config.json
├── environment.json
├── data_audit.json
├── partition_summary.json
├── metrics_by_round.csv
├── client_metrics.csv
├── final_metrics.json
├── confusion_matrix.csv
├── global_model.pt
├── run.log
└── checksums.sha256
```

`environment.json` tối thiểu có Python, Torch, Torchvision, Flower, CUDA, GPU/CPU, OS và commit hash.

Runner Flower hiện lưu history thực tế trong `metrics.json`; worker đồng thời chuẩn
hóa scalar theo round vào bảng `experiment_rounds`. Hai nơi phải giữ cùng số round;
payload JSON vẫn là bằng chứng đầy đủ cho per-class arrays và confusion matrix.
`client_metrics.json` và bảng `client_round_metrics` giữ metric/bytes theo từng
client-phase. `environment.json` được chụp tại thời điểm run và có checksum trong
`run_manifest.json`.

Communication được đo bằng tổng byte của Flower `ArrayRecord`, `MetricRecord` và
`ConfigRecord` thực sự gửi/nhận. Con số này không bao gồm framing của transport,
header mạng, retry hoặc TLS overhead; báo cáo phải dùng đúng tên phép đo này.

## 10. Bảng kết quả tối thiểu

### Bảng mô hình

| Scenario | Alpha | Accuracy | Macro F1 | Worst-client F1 | Time | Rounds | Total bytes |
|---|---:|---:|---:|---:|---:|---:|---:|

### Bảng từng lớp

| Scenario | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|

### Biểu đồ

- Macro F1 theo round.
- Loss theo round.
- Heatmap phân phối class/client.
- Confusion matrix.
- Accuracy/Macro F1 so sánh scenario.
- Total bytes hoặc time so với macro F1.

## 11. Diễn giải kết quả

- “FedProx cao hơn FedAvg ở alpha 0.1” chỉ có ý nghĩa trong config/dataset đã thử.
- Không viết “FedProx luôn tốt hơn”.
- Nếu centralized thấp bất thường, kiểm tra preprocessing/split trước khi kết luận.
- Nếu PlantVillage cho điểm rất cao, nhấn mạnh controlled background và cần in-the-wild validation.
- Nếu worst-client F1 thấp dù global F1 cao, đây là phát hiện quan trọng về fairness giữa cơ sở.

## 12. Checklist trước main run

Các ô đã đánh dấu dưới đây là cổng code/smoke đã có bằng chứng. Các ô liên quan dữ
liệu vẫn để trống cho đến khi chạy trên PlantVillage thật.

- [ ] Checksum và class count đã lưu.
- [ ] Data audit đạt; không có ảnh hỏng hoặc content-hash overlap train/test.
- [ ] Không overlap train/val/test.
- [ ] Heatmap partition đã duyệt.
- [x] Centralized pipeline học được trên một batch nhỏ.
- [x] Flower chạy được 1 round/4 client.
- [x] FedAvg aggregation test đạt.
- [x] FedProx loss test đạt.
- [x] Metric zero-division xử lý đúng.
- [ ] Config pilot đã khóa.
- [x] Artifact directory mới, không ghi đè.
- [x] Smoke result bị loại khỏi report exporter.
