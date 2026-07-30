# 08 — Ma trận truy vết yêu cầu → code → kiểm chứng

Trạng thái:

- **Done:** có code và kiểm thử/bằng chứng.
- **Partial:** có nền móng nhưng chưa kiểm tra end-to-end với ảnh thật.
- **Planned:** chưa code.

| Yêu cầu | Trạng thái | Code/tài liệu | Kiểm chứng hiện tại | Việc còn lại |
|---|---|---|---|---|
| Tên đề tài/class contract | Done | `constants.py`, `/project`, context doc | import/compile | giữ bất biến |
| IID partition | Done | `data/partitioning.py`, `data/profiles.py` | unit test complete/no duplicate + shared-split profile test | chạy PlantVillage |
| Dirichlet Non-IID | Done | `data/partitioning.py`, `data/profiles.py` | deterministic α=0.5/0.1 profile test | thử ảnh thật α 0.1 |
| PlantVillage manifest | Done | `data/manifest.py`, `data/profiles.py`, CLI | 18.160 ảnh tomato thật; IID/α=0.5/α=0.1 cùng 14.529 train và 3.631 test; hash khóa | tái tạo trên máy khác từ đúng source commit |
| Data integrity gate | Done | `data/audit.py`, CLI `audit-data` | cả ba audit PlantVillage passed, 0 error; corrupt/overlap tests đạt | theo dõi provenance khi source đổi |
| Global test trước partition | Done | CLI flow, `data/audit.py` | content-grouped split thật; 14 duplicate group không cắt qua train/test | giữ nguyên manifest/hash cho mọi run |
| Metric classification | Done | `ml/metrics.py`, `ml/reporting.py` | macro/per-class/group, harmful→healthy, spider F1, confusion tests + dashboard matrix | chạy PlantVillage |
| FedAvg aggregation | Done | `fl/aggregation.py`, `flower/server_app.py` | weighted/shape/NaN + Flower 4-client smoke | chạy ma trận PlantVillage thật |
| Synthetic FedAvg/FedProx | Done | `simulation.py` | end-to-end tests | không dùng làm result |
| MobileNetV2 | Done cho pilot | `ml/model.py` | pretrained PlantVillage pilot 1 epoch + versioned checkpoint/inference thật | chốt epoch/hyperparameter và main seeds |
| FedProx squared L2 | Done | `ml/trainer.py`, `flower/client_app.py` | known-value squared-L2 + Flower xác nhận `mu=0.01`; 158/314 tensor khác FedAvg trên smoke hai batch | chạy ma trận PlantVillage thật |
| Centralized baseline | Done cho pilot | `experiments/centralized.py`, `run_baseline_smoke.py` | PlantVillage pretrained 1 epoch: accuracy 0,9325, macro-F1 0,8688; checksum hợp lệ, pilot-only | khóa protocol và chạy main seeds |
| Local-only baseline | Done cho pilot | `experiments/local_only.py`, `run_baseline_smoke.py` | PlantVillage α=0.5 đủ 4 client/checkpoint; mean global Macro-F1 0,6313, worst 0,4653; pilot-only | khóa protocol và main seeds |
| Flower ClientApp | Done cho smoke | `flower/client_app.py`, `scripts/run_flower_smoke.py` | FedAvg/FedProx, 4/4 train và evaluate, 0 failure | chạy nhiều round trên PlantVillage |
| Flower ServerApp | Done cho smoke | `flower/server_app.py`, `flower/tracking.py`, `flower/smoke.py` | central/client metric, timing, communication, environment, checkpoint/checksum; 4-client run | main matrix PlantVillage |
| Checkpoint | Done | `ml/checkpoint.py`, centralized/local/server app | save/load/taxonomy/checksum + PlantVillage checkpoint SHA-256 và inference thật | giữ artifact bất biến |
| Communication metrics thật | Done | `flower/tracking.py`, server/DB/exporter | 109.596.948 payload bytes trên FedProx 1-round smoke; per-client/phase identities đầy đủ | báo cáo rõ không gồm transport/TLS overhead |
| API/database experiment lifecycle | Done cho local | `api/main.py`, `api/worker.py`, `api/results.py` | SQLite CRUD/worker smoke; PostgreSQL Compose readiness và API query thật | migration upgrade/rollback + worker profile trên PostgreSQL |
| Flower worker process boundary | Done cho local | `api/worker.py` | API chỉ queue; worker claim, audit, argv không shell, checkpoint/metric verify | recovery job treo + multi-worker locking stress |
| Dashboard | Done cho local demo | `frontend/*`, `/data-profiles` | form/status, metrics, curve, confusion matrix, class/client heatmap, login viewer/admin; Vite build | dữ liệu PlantVillage thật |
| Docker Compose | Done cho local demo | `compose.yaml`, Dockerfiles, Nginx config | build/up thật; `db/api/web` healthy; HTTP→HTTPS 308 và HTTPS health 200 | Flower worker profile + backup/restore/migration |
| Local inference | Done CLI | `ml/inference.py`, CLI `predict` | checkpoint PlantVillage + ảnh global-test thật; UTF-8 Windows regression | inference UI tại client |
| Research result exporter | Done | `experiments/export.py`, CLI | comparison/per-class/confusion/environment/checksum; synthetic exclusion tests | chạy trên main-study artifacts |
| Web TLS/API auth | Done cho local demo | `frontend/nginx.conf`, `api/auth.py`, `api/settings.py`, secret generator | TLS 1.2/1.3 + headers; thiếu token 401; viewer write 403; admin đi qua authorization | rate limit, rotation/revocation, trusted production CA/secret manager |
| Flower TLS/node auth | Planned | security doc, generated local credential material | key/CA generation unit test, chưa nối runtime | cấu hình và kiểm thử SuperLink/SuperNode nhiều máy |
| Secure aggregation/DP | Advanced | security doc | — | chỉ sau MVP |

## Kiểm thử đã chạy ở bản bàn giao

```text
PYTHONPATH=src .venv-flower/Scripts/python -m unittest discover -s tests -v
Result: 58 tests + 2 subtests, OK

python3 -m compileall -q src tests scripts
Result: OK

npm ci && npm run build
Result: Vite production build OK

docker compose up -d --build db api web
Result: db/api/web healthy; HTTP→HTTPS 308; HTTPS 200; 401/403 role gates đạt

PYTHONPATH=src python3 -m cropfed.cli demo --rounds 2 --local-epochs 1
Result: completed; artifact marked synthetic_smoke_only

.venv-flower/Scripts/python scripts/run_flower_smoke.py \
  --fixture-root artifacts/flower-proximal-exercise-20260730 --algorithm both
Result: FedAvg và FedProx đều đủ 4/4 train/evaluate, 0 failure;
checkpoint/checksum/class order hợp lệ; 158/314 tensor khác nhau;
summary marked research_result_valid=false

.venv-flower/Scripts/python scripts/run_api_worker_smoke.py \
  --output-root artifacts/api-worker-round-store-20260730 --algorithm fedprox
Result: API draft→queued→worker running→completed; audit/checkpoint/metrics đạt;
round 0–1 đọc từ normalized database store

.venv-flower/Scripts/python scripts/run_baseline_smoke.py \
  --output-root artifacts/baseline-image-smoke-20260730
Result: centralized + 4 local-only image checkpoints; audit đạt;
summary marked research_result_valid=false

.venv-flower/Scripts/python scripts/run_flower_smoke.py \
  --fixture-root artifacts/flower-environment-smoke-20260730 --algorithm fedprox
Result: 4/4 train/evaluate, environment checksum đạt, 8 client-phase entries,
109.596.948 payload bytes; research_result_valid=false
```

Chưa hoàn tất tại môi trường bàn giao:

- PlantVillage local-only/Flower pilot và main matrix nhiều seed (centralized pilot đã đạt);
- Flower worker profile/TLS node authentication nhiều máy vì chưa có dữ liệu thật và
  runtime SuperLink/SuperNode tách máy;
- migration upgrade/rollback và backup/restore PostgreSQL.

Flower/Ray trên Windows có ghi cảnh báo worker `access violation` trong lúc quản lý/thu hồi tiến trình, nhưng cả hai lượt trả mã 0 và có đủ artifact. Cảnh báo này được giữ trong smoke summary, không được xem là đã giải quyết cho production deployment.

Đây là giới hạn kiểm chứng, không phải khẳng định các đường chạy đó đã đạt.

Chi tiết provenance, split, checksum và centralized pilot nằm tại
[`11_PLANTVILLAGE_PILOT_REPORT.md`](11_PLANTVILLAGE_PILOT_REPORT.md).
