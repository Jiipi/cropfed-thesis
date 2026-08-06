# 06 — Lộ trình và bàn giao ngữ cảnh

## 1. Mốc hiện tại

Phiên bản nền `0.1.0` đã hoàn thành kiến trúc, baseline/Flower integration smoke ảnh,
metric theo round và client được chuẩn hóa vào database, communication payload thực,
exporter, dashboard, control plane API → database → worker → Flower và Compose local
`db/api/web` có HTTPS cùng bearer auth viewer/admin. PlantVillage tomato thật đã
được khóa provenance, tạo/audit đủ ba profile; centralized, local-only α=0.5, Flower
FedAvg IID một round và bốn Flower FedAvg/FedProx Non-IID (α=0.5/0.1) một round đều
đã chạy pilot trên ảnh thật và bị verify bởi `scripts/verify_plantvillage_pilots.py`.
Mọi artifact pilot bị khóa khỏi research export. Worker profile trên Docker Compose
đã claim được experiment từ PostgreSQL, chạy data audit POSIX và spawn Flower; Alembic
roundtrip và `pg_dump`/`pg_restore` đã được verify qua
`scripts/backup_postgres_volume.py`. Việc đúng tiếp theo là khóa `μ` trên validation
rồi khóa protocol; không mở rộng thuật toán.

## 2. Kế hoạch 14 tuần

| Tuần | Mục tiêu | Bằng chứng hoàn tất |
|---:|---|---|
| 1 | Khóa đề tài, câu hỏi nghiên cứu, phạm vi | `00_PROJECT_CONTEXT`, decision log |
| 2 | Dựng repository, core partition/aggregation/metric | unit test đạt |
| 3 | Chuẩn bị PlantVillage, scan/split/hash | manifest + class/overlap report |
| 4 | Centralized MobileNetV2 pilot | checkpoint + validation curve |
| 5 | Local-only baseline và inference local | 4 kết quả client + demo ảnh |
| 6 | Flower FedAvg IID 4 client | 1–5 round integration run |
| 7 | FedAvg Non-IID α=0.5/0.1 | heatmap + curves |
| 8 | FedProx và pilot μ | bảng validation μ |
| 9 | Nối Flower runner với API/database | dashboard thấy round/status thật |
| 10 | Khóa protocol, chạy main seed 2026 | artifact đầy đủ mọi scenario |
| 11 | Chạy seed 2027/2028 nếu tài nguyên | mean ± std |
| 12 | Phân tích, bảng và biểu đồ | notebook/script tái tạo figure |
| 13 | Bảo mật, Docker, integration test, demo | runbook + video/kịch bản demo |
| 14 | Hoàn thiện báo cáo, slide, đóng gói | report, source, model, hướng dẫn |

Nếu chỉ có 10–12 tuần, giảm seed/model phụ trước; không bỏ centralized/local-only hoặc Non-IID.

## 3. Backlog theo ưu tiên

### P0 — Làm ngay

1. [x] CPU QA runtime Torch/Torchvision và Flower/Ray simulation đã có; trước main run chỉ cần cài wheel đúng GPU nếu sử dụng CUDA.
2. [x] PlantVillage nguồn Mohanty đã checkout tại commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`; raw image không commit.
3. [x] `prepare-mvp-profiles` đã tạo IID/alpha 0.5/alpha 0.1 trên PlantVillage thật: 14.529 train, 3.631 shared global test.
4. [x] Ba audit PlantVillage đều đạt, 0 error; duplicate content được group trước split và không có train/test overlap.
5. [x] Centralized MobileNetV2 pretrained pilot một epoch hoàn tất; checkpoint/checksum hợp lệ, gắn `pilot_not_for_research`.
6. [x] Chạy Flower 1 round/4 client cho FedAvg/FedProx, lưu log/checkpoint và sửa integration error.
7. [x] Bổ sung local-only image runner và checkpoint version hóa.
8. [x] Local-only PlantVillage pilot bốn client trên α=0.5 đã hoàn tất, đủ checkpoint/hash và bị khóa pilot-only.
9. [x] Flower FedAvg IID PlantVillage pilot 1 round đã đủ 4/4 train/evaluate,
   0 failure; central Macro-F1 0,9145, checkpoint/environment hash hợp lệ và bị
   khóa pilot-only.
10. [x] Pilot FedAvg/FedProx trên α=0.5 và α=0.1 hoàn tất 4/4 pilot với evidence
    log + checksum + client_history đạt (`verify_plantvillage_pilots.py` `passed`).
    Chọn `μ` trên validation và khóa số round/epoch/batch vẫn là việc tiếp theo
    trước main study.

### P1 — Hoàn tất MVP

1. [x] Metric global/aggregate theo round có trong JSON/`experiment_rounds`; metric từng client-phase có trong `client_metrics.json` và `client_round_metrics`.
2. [x] Đã đo payload/model upload/download thực từ Flower Record, theo client/phase/round; phép đo ghi rõ không gồm transport/TLS overhead.
3. [x] API chỉ xếp hàng; worker riêng claim job và khởi chạy Flower bằng argv/profile whitelist.
4. [x] UI hiện metric Flower cuối, harmful→healthy, worst-client F1, communication, curve Macro-F1, confusion matrix và heatmap phân bố lớp/client.
5. [x] Inference local bằng CLI/checkpoint đã có; còn nút UI tại client nếu cần demo.
6. [x] Export comparison/per-class/confusion CSV, environment manifest, checksum; tự loại mọi synthetic smoke.
7. [x] API/SQLite lifecycle, whitelist, worker success/failure và API-worker-Flower smoke đã có.
8. [x] Compose local `db/api/web` build/up healthy; HTTPS, security headers và role gate 401/403 đã kiểm chứng.
9. [x] Flower worker profile trên PostgreSQL chạy end-to-end: API `start` → worker claim → audit → Flower spawn; Alembic roundtrip `base ↔ 0001_initial ↔ 0002_clients`; `scripts/backup_postgres_volume.py` đã verify restore schema 5 bảng. Còn recovery job treo + multi-worker locking stress + TLS/node auth.

### P2 — Chỉ sau MVP

- ResNet18;
- PlantWild;
- SCAFFOLD/FedAdam;
- secure aggregation/DP;
- multi-machine deployment.

## 4. Lệnh kiểm tra trước mỗi commit

```bash
make test
make compile
make smoke
make frontend-build
```

Khi đã cài full dependencies, bổ sung:

```bash
python -c "import torch, torchvision, flwr, fastapi, sqlmodel"
python scripts/run_flower_smoke.py --fixture-root artifacts/flower-smoke-local --algorithm both
python scripts/run_api_worker_smoke.py --output-root artifacts/api-worker-smoke-local --algorithm fedprox
python scripts/run_baseline_smoke.py --output-root artifacts/baseline-image-smoke-local
docker compose config
docker compose up --build
```

## 5. Handoff cho phiên làm việc mới

Thông tin cần đọc theo thứ tự:

1. `docs/00_PROJECT_CONTEXT.md`.
2. `docs/DECISION_LOG.md`.
3. `docs/08_TRACEABILITY_MATRIX.md`.
4. issue/việc đầu tiên chưa hoàn tất trong P0.
5. `git status`, test result và artifact mới nhất.

Khi kết thúc một phiên:

- cập nhật traceability/status;
- ghi decision mới;
- ghi test đã chạy;
- lưu lỗi đang chặn, command tái hiện và log ngắn;
- không chỉ để thông tin trong chat.

## 6. Sản phẩm cuối cùng

### Mã nguồn

- backend/API;
- Flower server/client;
- data preparation;
- centralized/local/FL runners;
- inference;
- frontend;
- tests và Compose.

### Artifact nghiên cứu

- manifest/checksum/partition summary;
- config và environment từng run;
- checkpoint;
- metric theo round/client;
- bảng, biểu đồ, confusion matrices;
- script tái tạo kết quả.

### Tài liệu

- báo cáo đồ án;
- tài liệu yêu cầu/kiến trúc;
- hướng dẫn cài/chạy;
- threat model;
- tài liệu API;
- kịch bản demo;
- slide;
- video demo nếu trường yêu cầu.

Raw dataset không nằm trong gói bàn giao; chỉ có hướng dẫn tải và sinh manifest.

## 7. Tiêu chí quyết định có làm nâng cao hay không

Chỉ mở P2 khi tất cả điều sau đúng:

- ma trận P0/P1 chạy được từ đầu đến cuối;
- kết quả tái lập tối thiểu một seed;
- không leakage;
- centralized/local/FedAvg/FedProx đều có;
- dashboard và inference demo được;
- báo cáo phương pháp đã có bản nháp.
