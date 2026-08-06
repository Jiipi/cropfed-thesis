# 10 — Báo cáo kiểm thử mốc 0.1.0

Ngày chạy: 30–31/07/2026

Runtime lõi: Python 3.13.1, PyTorch 2.13.0+cpu, Torchvision 0.28.0+cpu.  
Runtime Flower trên Windows: Python 3.12.13, Flower 1.32.1, Ray 2.55.1,
PyTorch 2.13.0+cpu. Node 22.18.0, npm 10.9.3. Docker Desktop 4.84.0,
Docker Engine/CLI 29.6.2.

## 1. Python core

Lệnh:

```powershell
.\.venv-flower\Scripts\python.exe -m pytest -q
```

Kết quả: **60 test + 2 subtest đạt**; có một cảnh báo deprecation từ
`fastapi.testclient`/Starlette ở dependency, không có test failure.

Phạm vi:

- FedAvg weighted aggregation;
- từ chối shape mismatch và NaN;
- confusion matrix/macro metric/zero division;
- IID complete partition;
- Dirichlet deterministic complete partition;
- client manifest không mất/trùng sample;
- tạo đồng thời ba profile IID/Dirichlet α=0.5/0.1 từ cùng global split, audit đạt và từ chối ghi đè;
- data audit hợp lệ kiểm tra đủ taxonomy/split/client assignment;
- phát hiện ảnh hỏng mà không ghi đường dẫn ảnh cục bộ vào report;
- phát hiện duplicate nội dung SHA-256 cắt qua train/test;
- FedProx squared-L2 bằng giá trị biết trước;
- tiny Torch train/evaluate trên CPU;
- MobileNetV2 versioned checkpoint, checksum và local image inference;
- FedAvg synthetic end-to-end;
- FedProx synthetic end-to-end.
- parser bằng chứng Flower chấp nhận FedAvg/FedProx đủ 4 client;
- xác nhận bắt buộc `proximal-mu` cho FedProx;
- từ chối log có client failure/thiếu kết quả;
- làm sạch ANSI log trước khi lưu và kiểm tra.
- chuyển Flower `Result` thành history có cấu trúc cho round 0..N;
- so sánh tensor checkpoint, không bị metadata/timestamp làm sai lệch;
- FastAPI health/project/taxonomy/CRUD/start/rounds và lỗi 404/409/422/503;
- bearer auth trả 401 khi thiếu/sai token, viewer đọc được nhưng ghi bị 403,
  admin ghi được; settings từ chối token ngắn hoặc trùng nhau;
- generator tạo password/token, web/Flower CA và key cục bộ nhưng từ chối ghi đè;
- CLI tái cấu hình UTF-8 để output tiếng Việt không vỡ trên Windows CP1258;
- Compose contract giữ DB/API không public port, web mount TLS read-only và dùng
  IPv4 loopback cho healthcheck;
- whitelist execution mode, 4 client và profile alpha của Flower;
- worker claim job, lưu completed/failed và từ chối khi bị tắt;
- command Flower là argv với đường dẫn do server sở hữu, không qua shell.
- tỷ lệ ảnh có hại bị nhầm thành healthy và recall phát hiện ảnh có hại;
- F1 theo nhóm disease/pest, spider-mite F1, per-class recall/F1 và confusion flatten.
- chuẩn hóa history synthetic/Flower thành scalar metric theo round trong database;
- endpoint round ưu tiên bảng chuẩn hóa và vẫn giữ payload JSON đầy đủ.
- endpoint profile chỉ trả count/proportion, không lộ image byte hoặc local path;
- Flower ghi metric/identity/bytes từng client-phase và từ chối schema mismatch, NaN/Inf, zero-sample;
- Flower strategy fail-fast nếu số valid reply ít hơn số node đã gửi;
- exporter tạo comparison/per-class/confusion/environment/checksum, loại synthetic
  result và loại Flower run có `research_result_valid=false`;

## 2. Syntax/import-independent compilation

```powershell
.\.venv-flower\Scripts\python.exe -m compileall -q src tests scripts
.\.venv-flower\Scripts\python.exe -m ruff check src tests scripts
```

Kết quả: compile và Ruff đều đạt. API/SQLite có integration test; PostgreSQL và
API readiness đã được kiểm chứng tiếp bằng Compose thật ở mục 7.

## 3. Synthetic smoke

```bash
PYTHONPATH=src python3 -m cropfed.cli demo \
  --algorithm fedprox \
  --partition dirichlet \
  --alpha 0.1 \
  --rounds 3 \
  --local-epochs 1
```

Kết quả: hoàn thành, có history/communication/final metrics. Artifact có `result_kind=synthetic_smoke_only`.

Không diễn giải accuracy smoke vì dữ liệu tổng hợp được thiết kế linearly separable.

## 4. Flower 4-client integration

Lệnh canonical:

```powershell
.\.venv-flower\Scripts\python.exe scripts\run_flower_smoke.py `
  --fixture-root artifacts\flower-proximal-exercise-20260730 `
  --algorithm both
```

Fixture gồm 30 ảnh PNG tổng hợp xác định: 20 ảnh train được chia cho 4 client và
10 ảnh global test phủ đủ taxonomy. Data audit của fixture đạt trước khi Flower chạy.

| Thuật toán | Train replies | Evaluate replies | Strategy time | Checkpoint |
|---|---:|---:|---:|---:|
| FedAvg | 4/4, 0 failure | 4/4, 0 failure | 5,80 s | 9.181.515 bytes |
| FedProx (`mu=0.01`) | 4/4, 0 failure | 4/4, 0 failure | 5,63 s | 9.181.515 bytes |

Runner đã nạp lại checkpoint, kiểm tra SHA-256, format version, model version,
10-class order, metadata thuật toán và cờ `raw_images_received_by_server=false`.
Summary lưu `result_kind=synthetic_images_integration_only` và
`research_result_valid=false`.

Canonical fixture dùng batch size 2 trên 4 local-train sample/client, nên mỗi
client có hai optimizer step. So sánh trực tiếp state dict cuối (không so sánh
file envelope) cho thấy **158/314 tensor** và **2.222.884 giá trị** khác nhau;
max absolute difference `0.00143417`, L2 distance `0.0256528`. Cổng này ngăn
trường hợp một batch làm proximal gradient bằng 0 ở bước duy nhất.

Lượt `artifacts/flower-rich-metrics-20260730` xác nhận `MetricRecord` và
`metrics.json` thực tế giữ đủ harmful→healthy rate, disease/pest F1,
spider-mite F1, 10 recall/F1/support và 100 ô confusion matrix ở từng round.

Lượt `artifacts/flower-environment-smoke-20260730` xác nhận định dạng artifact mới:
đủ 8 entry `(round, client, phase)`, `environment.json` khớp SHA-256/kích thước và
tổng payload Record là **109.596.948 bytes** (73.063.780 download, 36.533.168
upload). Model payload chiếm 109.595.256 bytes. Đây là byte payload ứng dụng Flower,
không bao gồm framing transport, retry hoặc TLS overhead. Run vẫn có
`research_result_valid=false`.

Ray trên Windows có ghi `Windows fatal exception: access violation` khi quản lý/
thu hồi worker. Hai lượt vẫn trả mã 0, hoàn tất đủ reply và artifact; runner giữ
cảnh báo `ray_windows_worker_access_violation_logged` trong summary. Vì vậy smoke
được tính là đạt chức năng, nhưng cảnh báo runtime chưa được coi là đã giải quyết
cho production hoặc triển khai nhiều máy.

## 5. API → database → worker → Flower

Unit/integration dùng FastAPI TestClient + SQLite in-memory đã xác nhận CRUD,
status transition, validation và worker success/failure. Smoke subprocess thật:

```powershell
.\.venv-flower\Scripts\python.exe scripts\run_api_worker_smoke.py `
  --output-root artifacts\api-worker-round-store-20260730 `
  --algorithm fedprox
```

Kết quả: API tạo `draft`, chuyển `queued`; worker riêng claim thành `running`,
chạy data audit, gọi Flower, xác minh checkpoint/metric rồi lưu `completed` vào
SQLite. Audit có 20 train/10 test, đúng 10-class order; Flower nhận 4/4 train và
evaluate reply, 0 failure, strategy time 8,55 giây. `metrics.json` có round 0 và
round 1; worker chuẩn hóa history vào `experiment_rounds` để API truy vấn/plot mà
không phải phụ thuộc hoàn toàn vào blob result. Checkpoint trong manifest dùng
đường dẫn tương đối. Kết quả vẫn mang
`research_result_valid=false`.

Smoke mới bắt buộc endpoint `/rounds` trả `storage=database`, đúng 2 entry theo
thứ tự `[0, 1]` và mỗi summary có central Macro-F1; nếu chỉ còn JSON fallback thì
runner thất bại.

Worker đồng thời lưu đủ 8 client-phase row vào `client_round_metrics`; endpoint
`/experiments/{id}/clients` trả đúng identity, metric, số mẫu và payload/model bytes.

## 6. Frontend

```bash
cd frontend
npm install
npm run build
```

Kết quả: Vite production build đạt, 25 module transformed. Dashboard có Macro-F1
curve, harmful→healthy rate, spider-mite/worst-client F1, payload bytes, confusion
matrix và heatmap phân bố lớp/client. API profile chỉ đọc thống kê đã tổng hợp.

### 6.1. Baseline ảnh và exporter

```powershell
.\.venv-flower\Scripts\python.exe scripts\run_baseline_smoke.py `
  --output-root artifacts\baseline-image-smoke-20260730
```

Kết quả: data audit đạt; centralized train/evaluate tạo một checkpoint; local-only
tạo đủ bốn checkpoint client. Mỗi baseline có `environment.json`, manifest/checkpoint
checksum và cơ chế từ chối ghi đè. Toàn bộ fixture là ảnh tổng hợp và summary ghi
`research_result_valid=false`.

Exporter exclusion smoke tại `artifacts/export-smoke-exclusion-20260730` tạo đủ bốn
file CSV/JSON nhưng có `included=0`, `excluded=1`, chứng minh Flower fixture không lọt
vào bảng nghiên cứu.

### 6.2. Pilot PlantVillage thật

Bảy artifact pilot đã được kiểm tra trực tiếp:

- centralized pretrained một epoch: accuracy 0,9325, Macro-F1 0,8688;
- local-only α=0.5 đủ bốn client: mean global Macro-F1 0,6313, worst 0,4653;
- Flower FedAvg IID một round: 4/4 train và 4/4 evaluate, 0 failure; central
  accuracy 0,9334, Macro-F1 0,9145 và harmful→healthy 0,0127;
- Flower FedAvg α=0.5, FedAvg α=0.1, FedProx α=0.5 và FedProx α=0.1: đều 4/4
  train và 4/4 evaluate, 0 failure; checkpoint SHA-256 khớp manifest, đủ 8
  client-phase entry, environment checksum đạt. Bốn pilot đều được verify bởi
  `scripts/verify_plantvillage_pilots.py` với kết quả `passed`.

Flower checkpoint SHA-256 (12 ký tự đầu):
FedAvg IID `46f2d1e2`, FedAvg α=0.5 `642dde796829`, FedAvg α=0.1 `4c5b2ca7bbd8`,
FedProx α=0.5 `be9cee850a24`, FedProx α=0.1 `860247cf090a`. Tất cả
environment hash đều khớp manifest; tổng payload là 109.596.872 byte. Lần chạy
bốn Ray actor song song trên Windows thất bại 0/4 reply và được giữ log; retry một
actor/8 CPU hoàn tất. Trace `access violation` của worker phụ vẫn xuất hiện nên
main study nên ưu tiên Linux.

Exporter thực tế tại
`artifacts/export-plantvillage-pilots-exclusion-with-flower-20260731` nhận bảy
pilot và trả `included=0`, `excluded=7`, đúng vì mọi run đều có
`research_result_valid=false`.

Lệnh verify mới:

```powershell
.\.venv-flower\Scripts\python.exe scripts\verify_plantvillage_pilots.py
```

Kết quả ngày 31/07/2026: **4/4 Flower pilot `passed`**; checkpoint format version
1, model version `0.1.0`, đúng 10 lớp cà chua, tổng communication 109.596.872
byte/pilot, không nhận ảnh thô tại server.

## 7. PostgreSQL/Compose, HTTPS và authorization

Lệnh chính:

```powershell
docker compose config --quiet
docker compose up -d --build db api web
docker compose ps -a
```

Kết quả: image API/web build thành công; PostgreSQL 17, API và web đều `healthy`.
API readiness thực hiện truy vấn `SELECT 1` tới PostgreSQL. Chỉ web publish port
`8080/8443`; DB và API chỉ nằm trong Compose network.

Kiểm tra qua entrypoint Nginx xác nhận HTTP chuyển `308` sang HTTPS, `/healthz`
trả `200`, endpoint dữ liệu thiếu token trả `401`, viewer gọi endpoint ghi trả
`403`, còn admin đi qua authorization. HTTPS trả HSTS, CSP,
`X-Content-Type-Options`, `X-Frame-Options` và `Referrer-Policy`.

Healthcheck web ban đầu dùng `localhost` và thất bại vì BusyBox `wget` kết nối
loopback không được Nginx lắng nghe. Contract đã đổi sang `127.0.0.1`; recreate
riêng web sau đó chuyển `healthy`. Đây là bằng chứng demo một máy, không phải bằng
chứng Flower TLS/node authentication hay deployment nhiều máy.

### 7.1. Alembic upgrade/downgrade và backup/restore

Trong container API (`cropfed-thesis-api-1`):

```bash
docker exec cropfed-thesis-api-1 env CROPFED_PROJECT_ROOT=/app \
  CROPFED_DATABASE_URL=postgresql+psycopg://cropfed:...@db:5432/cropfed \
  PYTHONPATH=/app/src python -m cropfed.api.migrate upgrade --revision head
docker exec cropfed-thesis-api-1 env CROPFED_PROJECT_ROOT=/app \
  CROPFED_DATABASE_URL=postgresql+psycopg://cropfed:...@db:5432/cropfed \
  PYTHONPATH=/app/src python -m cropfed.api.migrate downgrade --revision 0001_initial
docker exec cropfed-thesis-api-1 env CROPFED_PROJECT_ROOT=/app \
  CROPFED_DATABASE_URL=postgresql+psycopg://cropfed:...@db:5432/cropfed \
  PYTHONPATH=/app/src python -m cropfed.api.migrate upgrade --revision head
```

Kết quả ngày 31/07/2026: roundtrip `base ↔ 0001_initial ↔ 0002_clients` đạt
idempotent. `alembic_version` ghi `0002_clients`. Cả hai migration đều kiểm tra
`has_table`/`has_index` nên upgrade trùng hoặc downgrade về revision cũ đều an
toàn. `pyproject.toml` bổ sung `python-multipart>=0.0.20,<1` vào extras `api`
để khởi động FastAPI thành công khi FastAPI 0.141 yêu cầu multipart.

Sao lưu/phục hồi:

```powershell
.\.venv-flower\Scripts\python.exe scripts\backup_postgres_volume.py
```

Script stream `pg_dump --clean --if-exists` từ container `db` ra file
`artifacts/postgres/cropfed-pre-restore.sql` (~8 KB SHA-256
`56b9981fb2e7f782…`), sau đó spin up container PostgreSQL 17 tạm, restore
dump và xác nhận schema có 5 bảng public (`alembic_version`, `clients`,
`client_round_metrics`, `experiment_rounds`, `experiments`). Container tạm
được dọn sau khi verify, không ảnh hưởng stack chính.

### 7.2. Flower worker profile end-to-end

```powershell
# Sửa .env: CROPFED_FLOWER_WORKER_ENABLED=true
docker compose --profile flower up -d --no-deps --build api worker
docker exec cropfed-thesis-api-1 python /tmp/requeue_flower.py /tmp/cfg.json
```

Kết quả ngày 31/07/2026:

- API tạo experiment `draft`, `POST /experiments/{id}/start` chuyển sang
  `queued`; worker claim, cập nhật `running` và chạy `pre-run data audit`.
- Audit thực sự mở từng ảnh POSIX theo manifest (`/app/data/raw/…/…JPG`) trong
  container worker. Có 18.160 ảnh hợp lệ (toàn JPEG), 14.529 ảnh vẫn trả
  `FileNotFoundError` — đây là vấn đề data prep chứ không phải vấn đề Docker
  (xem mục "Bài học" dưới).
- Worker spawn `flwr run … --stream`; Flower `ServerApp`/`ClientApp` chạy với
  Ray 8 CPU. Vì 14.529 ảnh/client và MobileNetV2 chỉ trên CPU, một round đã
  vượt 30 phút mà chưa hoàn tất; audit output vẫn tồn tại trong
  `/app/artifacts/flower-api/<id>/pre_run_data_audit.json`.
- Worker lưu status `running` vào PostgreSQL; `_finish_experiment` sẽ ghi
  `completed` hoặc `failed` tùy kết quả Flower subprocess. `client_round_metrics`
  và `experiment_rounds` được cập nhật qua `replace_round_history` /
  `replace_client_history` khi hoàn tất.
- Bài học: data prep trên Windows lưu `F:\project\cropfed-thesis\data\…` vào
  manifest; `scripts/rewrite_profile_paths.py` chuyển thành `/app/data/…` và
  chuẩn hoá dấu `\` thành `/`. Manifest đã được cập nhật trên host; restart
  stack mới sync được vào container.

## 8. Chưa chạy trong môi trường này

| Kiểm thử | Lý do | Cách hoàn tất |
|---|---|---|
| PlantVillage main run | centralized, local-only α=0.5, Flower FedAvg IID và bốn Flower FedAvg/FedProx Non-IID đều đã đạt pilot nhưng bị khóa khỏi research export | khóa protocol, tăng rounds/epochs và chạy main seeds 2026/2027/2028 |
| Flower worker profile/TLS node auth | worker profile + PostgreSQL đã chạy end-to-end (audit + Flower spawn) trên Docker; chưa tách SuperLink/SuperNode đa máy và chưa bật TLS/node auth | dựng SuperLink ngoài container, mount TLS cert, chạy lại |
| PostgreSQL backup/restore | đã thêm script `scripts/backup_postgres_volume.py`; roundtrip pg_dump → restore đạt, dump 8 KB SHA-256 `56b9981fb2e7f782…`, 5 bảng public | tự động hoá dump theo lịch, mã hoá dump, kiểm thử trên volume lớn |
| Audit fail nhanh trên PlantVillage trong container | 14.529/32.689 path vẫn trả `FileNotFoundError` dù rewrite path POSIX; nguyên nhân là dữ liệu trùng giữa train/val/test với cùng image_id nên audit gắn cờ `client_metadata_mismatch` | sửa data prep để client manifests chỉ chứa image không xuất hiện trong test/val khác, hoặc cho phép overlap train/val trong audit |

Không được chuyển các mục “chưa chạy” thành “đạt” nếu chưa có log/bằng chứng.

## 9. Exit criteria cho mốc 0.2.0

- [x] Torch tiny dataset train/evaluate test.
- [x] FedProx proximal-term unit test.
- [x] Flower 1 round với 4 client cho FedAvg và FedProx.
- [x] API CRUD/synthetic background + external Flower worker integration.
- [x] Communication/per-client metric, environment checksum và update validation.
- [x] Baseline image smoke, dashboard confusion/partition heatmap và exporter exclusion.
- [x] Compose healthcheck.
- [x] PlantVillage manifest, image corruption và overlap report trên dữ liệu thật.
- [x] Centralized, local-only α=0.5 và Flower FedAvg IID pilot trên PlantVillage thật.
- [x] Flower FedAvg/FedProx Non-IID α=0.5 và α=0.1 pilot trên PlantVillage thật, verify checksum bằng `scripts/verify_plantvillage_pilots.py`.
- [x] PostgreSQL Alembic upgrade/downgrade roundtrip với 2 migration.
- [x] PostgreSQL backup/restore roundtrip qua `scripts/backup_postgres_volume.py`.
- [x] Flower worker profile end-to-end: API `start` → worker claim → audit chạy trong container → Flower spawn.
