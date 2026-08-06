# 08 — Ma trận truy vết yêu cầu → code → kiểm chứng

Trạng thái:

- **Done:** có code và kiểm thử/bằng chứng.
- **Partial:** có nền móng nhưng chưa kiểm tra end-to-end với ảnh thật.
- **Planned:** chưa code.

| Yêu cầu | Trạng thái | Code/tài liệu | Kiểm chứng hiện tại | Việc còn lại |
|---|---|---|---|---|
| Tên đề tài/class contract | Done | `constants.py`, `/project`, context doc | import/compile | giữ bất biến |
| IID partition | Done | `data/partitioning.py`, `data/profiles.py` | unit test + profile PlantVillage và Flower FedAvg IID pilot đạt | giữ manifest/hash bất biến cho main study |
| Dirichlet Non-IID | Done | `data/partitioning.py`, `data/profiles.py` | deterministic α=0.5/0.1 profile test | thử ảnh thật α 0.1 |
| PlantVillage manifest | Done | `data/manifest.py`, `data/profiles.py`, CLI | 18.160 ảnh tomato thật; IID/α=0.5/α=0.1 cùng 14.529 train và 3.631 test; hash khóa | tái tạo trên máy khác từ đúng source commit |
| Data integrity gate | Done | `data/audit.py`, CLI `audit-data` | cả ba audit PlantVillage passed, 0 error; corrupt/overlap tests đạt | theo dõi provenance khi source đổi |
| Global test trước partition | Done | CLI flow, `data/audit.py` | content-grouped split thật; 14 duplicate group không cắt qua train/test | giữ nguyên manifest/hash cho mọi run |
| Metric classification | Done | `ml/metrics.py`, `ml/reporting.py` | macro/per-class/group, harmful→healthy, spider F1, confusion tests + metric PlantVillage thật | dùng cùng contract cho main study |
| FedAvg aggregation | Done cho pilot | `fl/aggregation.py`, `flower/server_app.py` | weighted/shape/NaN + Flower IID PlantVillage 4 client, 1 round, 0 failure; Non-IID α=0.5/0.1 pilot với checksum và evidence log đạt | chạy ma trận Non-IID nhiều round/seed |
| Synthetic FedAvg/FedProx | Done | `simulation.py` | end-to-end tests | không dùng làm result |
| MobileNetV2 | Done cho pilot | `ml/model.py` | pretrained PlantVillage pilot 1 epoch + versioned checkpoint/inference thật | chốt epoch/hyperparameter và main seeds |
| FedProx squared L2 | Done | `ml/trainer.py`, `flower/client_app.py` | known-value squared-L2 + Flower xác nhận `mu=0.01`; 158/314 tensor khác FedAvg trên smoke hai batch; bốn pilot Non-IID PlantVillage checksum đạt | chạy ma trận PlantVillage thật |
| Centralized baseline | Done cho pilot | `experiments/centralized.py`, `run_baseline_smoke.py` | PlantVillage pretrained 1 epoch: accuracy 0,9325, macro-F1 0,8688; checksum hợp lệ, pilot-only | khóa protocol và chạy main seeds |
| Local-only baseline | Done cho pilot | `experiments/local_only.py`, `run_baseline_smoke.py` | PlantVillage α=0.5 đủ 4 client/checkpoint; mean global Macro-F1 0,6313, worst 0,4653; pilot-only | khóa protocol và main seeds |
| Flower ClientApp | Done cho pilot | `flower/client_app.py`, `scripts/run_flower_smoke.py`, `scripts/verify_plantvillage_pilots.py` | FedAvg/FedProx IID + Non-IID PlantVillage đủ 4/4 train/evaluate; central Macro-F1 IID 0,9145; Non-IID verification `passed` cho cả bốn pilot | chạy nhiều round và main seeds |
| Flower ServerApp | Done cho pilot | `flower/server_app.py`, `flower/tracking.py`, `flower/smoke.py`, `scripts/verify_plantvillage_pilots.py` | central/client metric, timing, communication, environment/checkpoint checksum; fail-fast nếu thiếu reply; bốn pilot Non-IID đều `aggregate_train: Received 4 results and 0 failures` | main matrix PlantVillage |
| Checkpoint | Done | `ml/checkpoint.py`, centralized/local/server app | save/load/taxonomy/checksum + PlantVillage checkpoint SHA-256 và inference thật | giữ artifact bất biến |
| Communication metrics thật | Done | `flower/tracking.py`, server/DB/exporter | PlantVillage FedAvg pilot: 109.596.872 payload byte; đủ 8 client-phase identity | báo cáo rõ không gồm transport/TLS overhead |
| API/database experiment lifecycle | Done cho local | `api/main.py`, `api/worker.py`, `api/results.py` | SQLite CRUD/worker smoke; PostgreSQL Compose readiness và API query thật; Alembic roundtrip `base ↔ 0001_initial ↔ 0002_clients`; Flower worker profile end-to-end (audit + Flower spawn) chạy với PostgreSQL | recovery job treo + multi-worker locking stress |
| Flower worker process boundary | Done cho local | `api/worker.py` | API chỉ queue; worker claim, audit, argv không shell, checkpoint/metric verify; Flower worker profile end-to-end chạy với PostgreSQL | recovery job treo + multi-worker locking stress |
| Dashboard | Done cho local demo | `frontend/*`, `/data-profiles` | form/status, metrics, curve, confusion matrix, class/client heatmap, login viewer/admin; Vite build | dữ liệu PlantVillage thật |
| Docker Compose | Done cho local demo | `compose.yaml`, Dockerfiles, Nginx config, `scripts/backup_postgres_volume.py` | build/up thật; `db/api/web/worker` healthy; HTTP→HTTPS 308 và HTTPS health 200; `pg_dump`/`pg_restore` roundtrip; Alembic upgrade/downgrade idempotent | Flower TLS/node auth + multi-machine |
| Local inference | Done CLI | `ml/inference.py`, CLI `predict` | checkpoint PlantVillage + ảnh global-test thật; UTF-8 Windows regression | inference UI tại client |
| Research result exporter | Done | `experiments/export.py`, CLI | comparison/per-class/confusion/environment/checksum; thực tế loại 3/3 PlantVillage pilot | chạy trên main-study artifacts |
| Web TLS/API auth | Done cho local demo | `frontend/nginx.conf`, `api/auth.py`, `api/settings.py`, secret generator | TLS 1.2/1.3 + headers; thiếu token 401; viewer write 403; admin đi qua authorization | rate limit, rotation/revocation, trusted production CA/secret manager |
| Flower TLS/node auth | Planned | security doc, generated local credential material | key/CA generation unit test, chưa nối runtime; worker profile chạy với PostgreSQL đã được kiểm chứng | cấu hình và kiểm thử SuperLink/SuperNode nhiều máy với TLS bật |
| Secure aggregation/DP | Advanced | security doc | — | chỉ sau MVP |

## Kiểm thử đã chạy ở bản bàn giao

```text
.venv-flower/Scripts/python -m pytest -q
Result: 60 tests + 2 subtests, OK

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

flwr run . local --stream  # FedAvg IID, 4 SuperNode, 1 actor/8 CPU, 1 round
Result: PlantVillage đủ 4/4 train và 4/4 evaluate, 0 failure;
central accuracy 0,9334, Macro-F1 0,9145, harmful→healthy 0,0127;
checkpoint/environment checksum đạt, 109.596.872 payload bytes;
artifact plantvillage-flower-fedavg-iid-pilot-seed2026 bị khóa pilot-only

.venv-flower/Scripts/python scripts/verify_plantvillage_pilots.py
Result: 4/4 PlantVilleage Flower pilot `passed`;
FedAvg/FedProx × α=0.5/α=0.1 đều có checkpoint SHA-256 khớp manifest,
client_history đủ 4 client × 1 round × 2 phase,
flower.log chứa `aggregate_train: Received 4 results and 0 failures`;
checkpoint format version 1, model version 0.1.0, đúng 10 lớp cà chua

docker compose up -d --build db api web
docker exec cropfed-thesis-api-1 env CROPFED_PROJECT_ROOT=/app \
  CROPFED_DATABASE_URL=postgresql+psycopg://cropfed:...@db:5432/cropfed \
  PYTHONPATH=/app/src python -m cropfed.api.migrate upgrade --revision head
docker exec cropfed-thesis-api-1 env CROPFED_PROJECT_ROOT=/app \
  CROPFED_DATABASE_URL=postgresql+psycopg://cropfed:...@db:5432/cropfed \
  PYTHONPATH=/app/src python -m cropfed.api.migrate downgrade --revision 0001_initial
docker exec cropfed-thesis-api-1 env CROPFED_PROJECT_ROOT=/app \
  CROPFED_DATABASE_URL=postgresql+psycopg://cropfed:...@db:5432/cropfed \
  PYTHONPATH=/app/src python -m cropfed.api.migrate upgrade --revision head
Result: alembic_version `0002_clients`; 5 bảng public; roundtrip idempotent

.\.venv-flower\Scripts\python.exe scripts\backup_postgres_volume.py
Result: pg_dump 8 KB SHA-256 56b9981fb2e7f782…; restore vào container tạm đạt
5 bảng public; container tạm được dọn

docker compose --profile flower up -d --no-deps --build api worker
docker exec cropfed-thesis-api-1 python /tmp/requeue_flower.py /tmp/cfg.json
Result: API tạo experiment draft, start → queued; worker claim → running;
audit chạy với 18.160 ảnh hợp lệ; Flower ServerApp/ClientApp spawn qua
Ray 8 CPU; audit artifact tại /app/artifacts/flower-api/<id>/pre_run_data_audit.json
```

Chưa hoàn tất tại môi trường bàn giao:

- PlantVillage FedAvg/FedProx Non-IID main matrix nhiều seed (pilot 4 pilots đã đạt,
  cần tăng rounds/epochs và 3 seed nghiên cứu chính 2026/2027/2028);
- Flower worker profile/TLS node authentication nhiều máy; worker profile + PostgreSQL
  đã chạy end-to-end (audit + Flower spawn) trong container, còn SuperLink/SuperNode
  tách máy và TLS;
- audit fail nhanh trên PlantVillage trong container (14.529 path trùng train/val/test
  kéo `client_metadata_mismatch`) — sửa data prep trước khi chạy Flower main matrix
  trong Docker.

Flower/Ray trên Windows: lần thử PlantVillage với bốn actor song song bị
`ActorDiedError` và 0/4 reply; log lỗi được giữ nguyên. Tracking strategy hiện
fail-fast khi thiếu valid reply. Lần chạy lại bằng một actor/8 CPU hoàn tất đủ
4/4 reply và artifact, dù Ray vẫn in trace `access violation` từ worker phụ lúc
khởi tạo/thu hồi. Đây là workaround local, không được xem là đã giải quyết cho
production deployment; ưu tiên Linux cho main study.

Đây là giới hạn kiểm chứng, không phải khẳng định các đường chạy đó đã đạt.

Chi tiết provenance, split, checksum và centralized pilot nằm tại
[`11_PLANTVILLAGE_PILOT_REPORT.md`](11_PLANTVILLAGE_PILOT_REPORT.md).
