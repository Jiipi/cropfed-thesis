# 09 — Dàn ý báo cáo đồ án

## Chương 1 — Tổng quan

- Bối cảnh sâu bệnh cây trồng và nhu cầu nhận dạng qua ảnh.
- Vấn đề dữ liệu phân tán giữa cơ sở.
- Lý do centralized khó áp dụng.
- Mục tiêu, RQ1–RQ6; ghi RQ6 là điều kiện nếu chưa chạy backbone đối chiếu.
- Đối tượng, phạm vi, đóng góp.
- Làm rõ “phát hiện” là classification ở mức ảnh.

## Chương 2 — Cơ sở lý thuyết

- Machine Learning, Deep Learning.
- CNN, transfer learning, MobileNetV2.
- Multiclass classification và metric.
- Federated Learning, cross-silo.
- IID/Non-IID, label skew, client drift.
- FedAvg.
- FedProx.
- Quyền riêng tư, secure aggregation, differential privacy.

## Chương 3 — Phân tích và thiết kế hệ thống

- Actor và use case.
- Functional/non-functional requirements.
- Kiến trúc web + experiment runner + Flower.
- Server/client responsibility.
- Sequence huấn luyện.
- Database/API.
- Threat model.
- Quyết định scope và phần ngoài phạm vi.

## Chương 4 — Dữ liệu và phương pháp thực nghiệm

- PlantVillage 38 lớp trên 14 loại cây (mốc 0.1.0 dùng 10 lớp cà chua — ghi rõ vì
  checkpoint và artifact giai đoạn đó thuộc taxonomy khác).
- Scan/manifest/data quality.
- Split 64/16/20 và chống leakage.
- Mô phỏng 4 client.
- Sáu profile: IID, Dirichlet alpha 100/0.5/0.1, quantity skew, feature skew — tất cả
  dùng chung một global test set.
- MobileNetV2 và hyperparameter.
- Centralized/local/FedAvg/FedProx/FedBN/SCAFFOLD/MOON.
- Protocol, seed, hardware, artifact.
- Metric phân loại và hệ thống; fairness giữa client (std/spread, không chỉ sàn) và
  khoảng cách vs centralized với quy ước dấu ghi rõ.

## Chương 5 — Cài đặt

- Cấu trúc source.
- Data preparation.
- PyTorch trainer.
- Flower ClientApp/ServerApp.
- API, database, dashboard.
- Docker.
- Test và reproducibility.

Không dán toàn bộ source code; chỉ trích đoạn quan trọng như partition, aggregation và FedProx objective.

## Chương 6 — Kết quả và thảo luận

- Phân phối class/client.
- Centralized vs local-only vs FL.
- Alpha và client drift.
- FedAvg vs FedProx.
- Per-class/confusion matrix.
- Worst-client fairness.
- Time/round/bytes.
- Kết quả nhiều seed.
- Giải thích bất thường.
- Hạn chế PlantVillage và simulation.

## Chương 7 — Kết luận và hướng phát triển

- Trả lời từng RQ bằng bằng chứng.
- Đóng góp.
- Hạn chế.
- PlantWild/field data.
- Secure aggregation/DP.
- Multi-machine, dropout.
- Detection/segmentation như hướng riêng, không giả vờ đã làm.

## Phụ lục

- Class mapping.
- Experiment matrix.
- API endpoints.
- Config đầy đủ.
- Environment/dependency.
- Hướng dẫn tái lập.
- Link source và checksum artifact.

## Bảng/figure cần chuẩn bị

1. Kiến trúc tổng thể.
2. Sequence một FL round.
3. Heatmap class/client cho IID, alpha 0.5, alpha 0.1.
4. Learning curves.
5. Confusion matrices.
6. Bảng metric tổng.
7. Bảng per-class.
8. Trade-off F1–time/bytes.
9. Screenshot dashboard.
