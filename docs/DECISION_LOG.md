# Decision log

Tài liệu này ghi các quyết định ảnh hưởng đề tài. Trạng thái `Accepted` chỉ thay đổi bằng một entry mới, không xóa lịch sử.

| ID | Ngày | Trạng thái | Quyết định | Lý do chính |
|---|---|---|---|---|
| D-001 | 2026-07-30 | Accepted | Giữ nguyên tuyệt đối tên đề tài chính thức | yêu cầu hành chính của đồ án |
| D-002 | 2026-07-30 | Accepted | “Phát hiện” = classification ở mức ảnh trong MVP | phù hợp dữ liệu/nguồn gần bài toán; object detection cần annotation khác |
| D-003 | 2026-07-30 | Accepted | Dùng subset cà chua PlantVillage 10 lớp | scope vừa sức, có healthy/disease/pest |
| D-004 | 2026-07-30 | Accepted | Bốn client cross-silo mô phỏng | đại diện cơ sở, đủ quan sát heterogeneity, phù hợp đồ án |
| D-005 | 2026-07-30 | Accepted | IID + Dirichlet alpha 0.5/0.1 | có baseline và hai mức label skew |
| D-006 | 2026-07-30 | Accepted | MobileNetV2 là model chính | cân bằng compute/communication; có transfer learning |
| D-007 | 2026-07-30 | Accepted | Bắt buộc local-only, centralized, FedAvg, FedProx | trả lời đầy đủ giá trị hợp tác và ảnh hưởng Non-IID |
| D-008 | 2026-07-30 | Accepted | Flower Message API >=1.32,<2 | tài liệu hiện hành; tránh bắt đầu bằng legacy API |
| D-009 | 2026-07-30 | Accepted | FastAPI + React + PostgreSQL; image training ở runner riêng | tách control plane khỏi compute dài |
| D-010 | 2026-07-30 | Accepted | Tách global test trước client partition | ngăn leakage và giữ comparison công bằng |
| D-011 | 2026-07-30 | Accepted | Macro F1 là metric chính | class imbalance; accuracy không đủ |
| D-012 | 2026-07-30 | Accepted | Smoke NumPy phải có nhãn `synthetic_smoke_only` | kiểm tra code nhưng ngăn nhầm thành kết quả nghiên cứu |
| D-013 | 2026-07-30 | Accepted | Raw images không nằm trong source/bàn giao | license, kích thước và privacy boundary |
| D-014 | 2026-07-30 | Accepted | FL không được mô tả là bảo đảm privacy hoàn chỉnh | update/model vẫn có nguy cơ rò rỉ |
| D-015 | 2026-07-30 | Accepted | Dùng RQ1–RQ5 cho lõi MVP; RQ6 về mô hình nhẹ chỉ kết luận đầy đủ khi có backbone đối chiếu | giữ đúng mục tiêu nghiên cứu nhưng không biến giả thuyết MobileNetV2 thành kết luận thiếu đối chứng |
| D-016 | 2026-07-30 | Accepted | Checkpoint dùng envelope có format/model version, class order, metadata và checksum; loader vẫn hỗ trợ raw state dict cũ | ngăn nhầm taxonomy/model, tăng truy vết và không phá checkpoint 0.1.0 |
| D-017 | 2026-07-30 | Accepted | Trên Windows, dùng virtualenv Python 3.12 riêng cho Flower/Ray simulation đã kiểm chứng; môi trường lõi vẫn có thể dùng Python 3.13 | Ray 2.55.1 có runtime Windows tương thích với Python 3.12 trong môi trường kiểm chứng, còn Python 3.13 không cài được simulation extra đầy đủ |
| D-018 | 2026-07-30 | Accepted | Flower integration smoke phải lưu log, checkpoint, checksum và tự xác nhận 4/4 client; artifact ảnh tổng hợp luôn mang `research_result_valid=false` | tạo bằng chứng end-to-end có thể kiểm tra nhưng ngăn nhầm smoke thành kết quả nghiên cứu |
| D-019 | 2026-07-30 | Accepted | FastAPI chỉ tạo/xếp hàng Flower job; worker riêng đọc DB và chạy subprocess bằng argv cùng profile dữ liệu phía server | không chạy image training dài trong web worker và không cho HTTP truyền shell command/đường dẫn dữ liệu |
| D-020 | 2026-07-30 | Accepted | Canonical FedProx smoke phải có ít nhất hai local optimizer step và so sánh tensor với FedAvg | một batch duy nhất làm proximal gradient bằng 0 tại bước đầu, chỉ chứng minh wiring chứ chưa chứng minh hạng proximal tác động |
| D-021 | 2026-07-30 | Accepted | Giữ artifact history JSON đầy đủ đồng thời chuẩn hóa scalar mỗi round vào `experiment_rounds` | JSON bảo toàn per-class/confusion để tái lập; cột scalar giúp API/dashboard truy vấn ổn định mà không làm mất bằng chứng gốc |
| D-022 | 2026-07-30 | Accepted | Đo communication bằng byte thực của Flower Array/Metric/Config Record theo từng chiều, phase và client; tên phép đo luôn ghi rõ loại trừ transport/TLS overhead | có số đo kiểm chứng được từ payload ứng dụng nhưng không tuyên bố sai là toàn bộ network traffic |
| D-023 | 2026-07-30 | Accepted | Exporter luôn loại artifact tổng hợp hoặc có `research_result_valid=false`; output mới có checksum và environment manifest | ngăn smoke metric lọt vào bảng kết quả nghiên cứu và tăng khả năng truy vết |
| D-024 | 2026-07-30 | Accepted | Ba profile IID/Dirichlet dùng cùng một split train/global-test được tạo trong một lệnh, xác nhận bằng SHA-256 và không ghi đè output cũ | tránh vô tình đổi test set giữa thuật toán và giảm sai thao tác thủ công |
| D-025 | 2026-07-30 | Accepted | Dashboard chỉ nhận class count/proportion từ `partition_summary.json`, không nhận byte ảnh hoặc local path | đáp ứng thống kê/heatmap theo cơ sở mà giữ ranh giới dữ liệu thô tại client |
| D-026 | 2026-07-30 | Accepted | Flower server từ chối model update sai schema, NaN/Inf, zero-sample, duplicate identity hoặc metadata quá giới hạn trước aggregation | áp dụng NFR kiểm tra update tại đúng trust boundary, không chỉ ở lõi NumPy test |
| D-027 | 2026-07-30 | Accepted | Demo Compose bắt buộc dùng secret sinh ngẫu nhiên, bearer token tách vai trò viewer/admin và HTTPS tại Nginx; DB/API không publish port | đóng ranh giới điều khiển local mà không đưa credential vào frontend image; không ngầm coi đây là Flower node auth hay bảo mật production nhiều máy |

## Mẫu entry mới

```text
ID:
Ngày:
Trạng thái: Proposed | Accepted | Superseded
Bối cảnh:
Quyết định:
Các phương án đã cân nhắc:
Hệ quả:
Files/tests cần cập nhật:
```
