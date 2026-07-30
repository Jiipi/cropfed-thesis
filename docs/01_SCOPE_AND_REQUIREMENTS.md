# 01 — Phạm vi và yêu cầu

## 1. Đối tượng sử dụng

| Đối tượng | Nhu cầu chính | Quyền trong MVP |
|---|---|---|
| Sinh viên/nghiên cứu viên | cấu hình, chạy, theo dõi và so sánh thí nghiệm | tạo/chạy/xem experiment; xuất metric |
| Quản trị coordinator | đăng ký client, kiểm tra trạng thái federation, lưu checkpoint | cấu hình server, xem log hệ thống |
| Người vận hành cơ sở nông nghiệp | trỏ tới dữ liệu cục bộ, tham gia train/evaluate, dự đoán ảnh tại chỗ | chạy client; không xem dữ liệu cơ sở khác |
| Giảng viên/hội đồng | kiểm tra phương pháp, tính tái lập và bằng chứng | quyền đọc dashboard, config, biểu đồ, báo cáo |
| Cán bộ kỹ thuật nông nghiệp | dùng mô hình tham khảo để nhận dạng một ảnh | inference cục bộ; thấy xác suất/cảnh báo giới hạn |

Không xây vai trò “bác sĩ cây trồng tự động”. Kết quả chỉ hỗ trợ sàng lọc, không thay thế chuyên gia.

## 2. Phạm vi bắt buộc và nâng cao

### Bắt buộc cho MVP

- Một cây trồng: cà chua.
- Mười lớp đã khóa trong `cropfed.constants.TOMATO_CLASSES`.
- Bốn client mô phỏng.
- IID, Dirichlet `α=0.5`, Dirichlet `α=0.1`.
- MobileNetV2.
- Centralized, local-only, FedAvg, FedProx.
- Dashboard quản lý experiment và xem metric.
- Inference một ảnh tại client.
- Báo cáo so sánh accuracy/macro metrics, thời gian, round, giao tiếp.

### Nâng cao

- ResNet18, SCAFFOLD, FedAdam/FedYogi.
- Secure aggregation hoặc differential privacy.
- Dữ liệu in-the-wild.
- Client dropout, nhiều máy thật, monitoring nâng cao.

## 3. Yêu cầu chức năng

| Mã | Yêu cầu | Tiêu chí chấp nhận MVP |
|---|---|---|
| FR-01 | Quản lý taxonomy | hệ thống trả đúng 10 ID/nhãn và checkpoint dùng cùng thứ tự |
| FR-02 | Kiểm kê dữ liệu | scan đủ thư mục; từ chối thiếu lớp, file không hợp lệ hoặc manifest rỗng |
| FR-03 | Tách dữ liệu | global test không giao với train/client validation |
| FR-04 | Phân client | tạo 4 partition IID hoặc Dirichlet, không mất/trùng index, tái lập theo seed |
| FR-05 | Cấu hình experiment | lưu algorithm, alpha, round, local epoch, LR, batch, seed, model |
| FR-06 | Huấn luyện cục bộ | client đọc ảnh local, nhận global weights, train và trả update + số mẫu |
| FR-07 | FedAvg | server tổng hợp update theo trọng số số mẫu |
| FR-08 | FedProx | client thêm `μ/2 · ||w-w_t||²`; server tổng hợp như FedAvg |
| FR-09 | Đánh giá | trả accuracy, macro P/R/F1, per-class metric, confusion matrix |
| FR-10 | Đo hệ thống | ghi thời gian round/run, số round, byte upload/download |
| FR-11 | Checkpoint | lưu global model, model/format version, config, seed, class order, timestamp và SHA-256 |
| FR-12 | Dashboard | xem experiment, trạng thái, metric theo round và kết quả cuối |
| FR-13 | Baseline | chạy centralized và local-only trên split tương thích |
| FR-14 | Inference local | client chọn ảnh, nhận crop, nhãn, nhóm khỏe/bệnh/sâu hại, top-k, confidence và model version mà không upload ảnh |
| FR-15 | Audit | mỗi kết quả truy ngược được về config và phiên bản code |

## 4. Yêu cầu phi chức năng

| Mã | Nhóm | Yêu cầu có thể kiểm tra |
|---|---|---|
| NFR-01 | Riêng tư | request từ client tới server không chứa path hoặc byte ảnh |
| NFR-02 | Bảo mật | deployment nhiều máy dùng TLS và xác thực node; secret không commit |
| NFR-03 | Tái lập | seed, split, config, dependency và class order được lưu |
| NFR-04 | Tin cậy | update sai key/shape, NaN/Inf hoặc zero sample bị từ chối |
| NFR-05 | Tính đúng | unit test cho partition, aggregation, metric và smoke pipeline |
| NFR-06 | Khả dụng | dashboard dùng được ở desktop và màn hình nhỏ; lỗi có thông báo |
| NFR-07 | Hiệu năng | model chính phù hợp GPU phổ thông; batch/worker cấu hình được |
| NFR-08 | Khả chuyển | chạy local bằng Python; web stack chạy bằng Compose |
| NFR-09 | Duy trì | code chia module data/ML/FL/API; quyết định được ghi tài liệu |
| NFR-10 | Quan sát | trạng thái draft/queued/running/completed/failed và log theo run |

## 5. Quy tắc nghiệp vụ

1. Một experiment đã chạy không được sửa config; muốn đổi phải tạo experiment mới.
2. Seed và class order là phần của danh tính experiment.
3. Test set chỉ dùng để báo cáo sau khi đã khóa model/hyperparameter.
4. FedAvg/FedProx phải dùng cùng initialization và split khi so sánh.
5. Metric tổng hợp không thay thế metric từng client/từng lớp.
6. `synthetic_smoke_only` không được xuất sang bảng kết quả nghiên cứu.
7. Server không có endpoint upload ảnh huấn luyện.

## 6. Use case chính

### UC-01 — Chuẩn bị dữ liệu

Người nghiên cứu chỉ định thư mục PlantVillage → hệ thống kiểm tra 10 lớp → tách global test → phân phần train thành bốn client → tạo manifest và thống kê nhãn.

### UC-02 — Chạy FL

Người nghiên cứu tạo config → coordinator tải initial model → mỗi client train local → coordinator tổng hợp → client/global evaluation → lưu metric → lặp đến hết round → lưu checkpoint.

### UC-03 — So sánh

Người nghiên cứu chọn các run cùng seed/split → hệ thống hiển thị centralized/local/FedAvg/FedProx theo macro F1, worst-client F1, thời gian và byte.

### UC-04 — Dự đoán tại cơ sở

Người vận hành chọn ảnh local → client tiền xử lý → model local suy luận → hiển thị top-k → không gửi ảnh về coordinator.

## 7. Definition of Done cho MVP

- Tất cả FR-01 đến FR-15 có test hoặc bằng chứng demo.
- Chạy thành công ma trận bắt buộc ít nhất một seed; bản báo cáo ưu tiên ba seed.
- Không có giao nhau train/validation/test.
- Có bảng phân phối nhãn từng client cho mọi alpha.
- Có bảng centralized/local/FedAvg/FedProx.
- Có confusion matrix và per-class recall/F1.
- Có đo communication và training time.
- Có demo dashboard + inference.
- Có hướng dẫn tái lập từ môi trường sạch.
- Báo cáo nêu rõ PlantVillage có nền ảnh kiểm soát và FL không đồng nghĩa privacy hoàn chỉnh.
