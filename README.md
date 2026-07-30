# CropFed Thesis

## Tên đề tài chính thức

**“Nghiên cứu và xây dựng hệ thống học liên kết (Federated Learning) cho phát hiện sâu bệnh cây trồng qua ảnh trên dữ liệu phân tán, không đồng nhất giữa các cơ sở nông nghiệp.”**

Tên trên là bất biến. Trong phạm vi triển khai, “phát hiện qua ảnh” được định nghĩa là **nhận dạng/phân loại đa lớp ở mức toàn ảnh** (image-level multiclass classification), không phải khoanh vùng bằng bounding box hay segmentation.

## Dự án này đang có gì?

Phiên bản `0.1.0` là nền móng chạy được để bắt đầu code mà không làm mất các quyết định nghiên cứu:

- Lõi NumPy mô phỏng đầy đủ IID/Non-IID, local training, FedAvg, FedProx, metric và chi phí giao tiếp.
- Bộ tạo manifest cho 10 lớp cà chua của PlantVillage; lệnh `prepare-mvp-profiles` tạo và audit đồng thời IID, Dirichlet α=0.5 và α=0.1 từ đúng một global split; ảnh gốc không được sao chép vào repository.
- PlantVillage tomato thật đã được khóa ở source commit `7f7ecc7e...`: 18.160 ảnh,
  14.529 train/3.631 global test, cả ba profile audit đạt và không leakage theo
  content hash.
- Cổng kiểm định dữ liệu cục bộ: giải mã ảnh, SHA-256, taxonomy, duplicate và overlap train/test/client.
- MobileNetV2 transfer learning; ResNet18 là mô hình đối chiếu.
- Checkpoint có version, class order, metadata và SHA-256; vẫn đọc được raw state dict cũ.
- Flower Message API với `ClientApp` và `ServerApp`; FedAvg/FedProx đã chạy end-to-end với 4 client trên fixture ảnh tổng hợp.
- Centralized và bốn local-only baseline đã chạy qua integration smoke ảnh; mỗi run có checkpoint, checksum và environment manifest.
- Centralized MobileNetV2 pretrained pilot một epoch đã chạy trên PlantVillage thật;
  checkpoint/inference hợp lệ nhưng artifact được đánh dấu pilot-only.
- Local-only α=0.5 pilot đã hoàn tất đủ bốn client; mean/worst global Macro-F1 cùng
  bốn checkpoint được lưu nhưng vẫn bị loại khỏi research export.
- FastAPI + PostgreSQL lưu metadata, scalar metric theo round và metric theo từng client; worker riêng claim Flower job bằng cấu hình/profile dữ liệu đã whitelist.
- Flower ghi thời gian, payload/model upload/download thực theo từng phase/client; phép đo loại trừ framing mạng và TLS overhead.
- React dashboard tạo synthetic/Flower job, đọc metric có cấu trúc, vẽ Macro-F1, confusion matrix và heatmap phân bố lớp/client mà không đọc ảnh/path cục bộ.
- Exporter sinh bảng comparison/per-class/confusion cùng environment/checksum và luôn loại smoke tổng hợp.
- Docker Compose cho web/API/database và Flower worker tùy chọn; entrypoint web dùng
  HTTPS, API dùng bearer token phân quyền `viewer`/`admin`.
- 58 kiểm thử tự động, Torch CPU runtime, baseline ảnh, Flower 4-client smoke,
  API-worker smoke, Compose healthcheck và frontend production build đã đạt ở các
  lần kiểm chứng.

> **Cảnh báo khoa học:** kết quả trong `artifacts/smoke-result.json` và fixture do
> `run_flower_smoke.py` tạo là dữ liệu tổng hợp. Centralized PlantVillage một epoch
> cũng chỉ là pilot. Tất cả đều khóa `research_result_valid=false`; tuyệt đối không
> dùng các metric này làm kết quả nghiên cứu chính.

## Bắt đầu nhanh

### 1. Kiểm tra lõi không cần dataset/GPU

```bash
cd cropfed-thesis
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m cropfed.cli demo \
  --algorithm fedavg \
  --partition dirichlet \
  --alpha 0.5 \
  --clients 4 \
  --rounds 5
```

### 2. Cài môi trường đầy đủ

Python 3.11–3.13:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[api,dev]"
```

PyTorch có nhiều biến thể CUDA. Nếu máy dùng GPU, nên cài wheel phù hợp theo [hướng dẫn chính thức của PyTorch](https://pytorch.org/get-started/locally/) trước, rồi mới chạy lệnh trên.

Riêng Flower simulation trên Windows đã được kiểm chứng bằng Python 3.12.13, Flower 1.32.1 và Ray 2.55.1. Nếu môi trường chính dùng Python 3.13, hãy tạo một virtualenv Python 3.12 riêng cho simulation; điều này không thay đổi contract Python của mã nguồn.

### 3. Chuẩn bị PlantVillage

Thư mục đầu vào phải chứa đúng 10 thư mục lớp ghi trong [tài liệu dữ liệu](docs/03_DATA_AND_NON_IID.md). Lệnh chuẩn sau tách global test đúng một lần, tạo ba profile mà worker nhận diện và tự audit từng profile:

```bash
cropfed prepare-mvp-profiles \
  --dataset-root /duong-dan/toi/plantvillage/color \
  --output-root data/flower-profiles \
  --clients 4 \
  --seed 2026
```

Kết quả gồm `iid`, `dirichlet-alpha-0.5`, `dirichlet-alpha-0.1` và
`profiles_index.json`. Hash train/global-test phải giống nhau giữa ba profile; chỉ
manifest tại `clients/` thay đổi. Lệnh từ chối ghi đè một profile set không rỗng.

Nếu chỉ cần tạo thủ công một profile:

```bash
cropfed prepare-data \
  --dataset-root /duong-dan/toi/plantvillage/color \
  --client-output-root data/clients_alpha05 \
  --clients 4 \
  --partition dirichlet \
  --alpha 0.5 \
  --seed 2026
```

Ảnh vẫn ở đường dẫn nguồn; manifest chỉ giữ đường dẫn và nhãn.

Kiểm định dữ liệu trước khi huấn luyện:

```bash
cropfed audit-data \
  --train-manifest data/processed/train_manifest.csv \
  --test-manifest data/processed/test_manifest.csv \
  --client-data-root data/clients_alpha05 \
  --clients 4 \
  --output data/processed/data_audit_alpha05.json
```

Lệnh trả mã khác 0 nếu có ảnh hỏng, sai taxonomy, giao nhau theo ID/path/hash
hoặc client partition bị mất/trùng mẫu. JSON chỉ chứa ID, hash, count và mã lỗi;
không chứa byte ảnh hay đường dẫn ảnh cục bộ.

Worker không nhận đường dẫn từ HTTP. Ba profile dữ liệu phải nằm theo tên cố định:

```text
data/flower-profiles/
├── iid/
├── dirichlet-alpha-0.5/
└── dirichlet-alpha-0.1/
```

Mỗi profile chứa `train_manifest.csv`, `test_manifest.csv`, `clients/client_0..3`
(hoặc hai manifest trong thư mục con `processed/`). Có thể tạo từng profile bằng
`cropfed prepare-data --output-dir <profile> --client-output-root <profile>/clients ...`;
worker luôn chạy lại data audit trước khi gọi Flower.
Dashboard chỉ đọc `partition_summary.json` để hiển thị count/heatmap; endpoint này
không trả byte ảnh hoặc đường dẫn ảnh cục bộ.

Worker mặc định bị tắt. Compose local bật bearer authentication với hai vai trò:
`viewer` chỉ đọc và `admin` được tạo/chạy thí nghiệm. Nginx kết thúc TLS; frontend
chỉ giữ token trong `sessionStorage`, không đóng token vào image. Đây vẫn là cấu
hình demo một máy: TLS/xác thực giữa Flower SuperNode và SuperLink chưa được nối
vào runtime, nên không expose worker ra Internet hoặc mạng không tin cậy.

### 4. Chạy centralized baseline

```bash
cropfed train-centralized \
  --model mobilenet_v2 \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.001
```

Chạy bốn mô hình local-only trên đúng profile và global test tương ứng:

```bash
cropfed train-local-only \
  --client-data-root data/flower-profiles/dirichlet-alpha-0.5/clients \
  --test-manifest data/flower-profiles/dirichlet-alpha-0.5/test_manifest.csv \
  --partition dirichlet \
  --alpha 0.5 \
  --clients 4 \
  --output-dir artifacts/local-only-alpha05-seed2026
```

Smoke riêng cho hai baseline (ảnh tổng hợp, không phải kết quả nghiên cứu):

```bash
python scripts/run_baseline_smoke.py \
  --output-root artifacts/baseline-image-smoke-local
```

Xuất các run đã duyệt sang CSV; run có `research_result_valid=false` luôn bị loại:

```bash
cropfed export-results \
  --run artifacts/centralized-seed2026 \
  --run artifacts/local-only-alpha05-seed2026 \
  --run artifacts/flower/fedavg-alpha-0.5-seed-2026 \
  --output-dir artifacts/report-export-seed2026
```

### 5. Dự đoán cục bộ

```bash
cropfed predict \
  --checkpoint artifacts/centralized/centralized_model.pt \
  --image /duong-dan/toi/anh-la.jpg
```

Checkpoint mới tự mang tên kiến trúc và model version. Kết quả gồm cây trồng,
nhãn/nhóm dự đoán, confidence, top-k, model version và cảnh báo giới hạn.

### 6. Chạy Flower local simulation

Trước khi dùng dữ liệu thật, chạy integration smoke tự kiểm tra cho cả FedAvg và FedProx:

```bash
python scripts/run_flower_smoke.py \
  --fixture-root artifacts/flower-smoke-local \
  --algorithm both
```

Runner tạo 30 ảnh tổng hợp cục bộ, audit split, chạy 4 client, lưu log/checkpoint/metric theo round và chỉ báo đạt khi đủ 4/4 kết quả train/evaluate, checksum hợp lệ và `proximal-mu` của FedProx được xác nhận. Fixture dùng hai local batch; khi chạy `both`, runner còn so sánh state dict và từ chối nếu FedAvg/FedProx không tạo cập nhật khác nhau. Dùng `--verify-existing` để kiểm tra lại artifact mà không khởi chạy Flower.

Smoke toàn bộ control plane API → SQLite → worker → Flower:

```bash
python scripts/run_api_worker_smoke.py \
  --output-root artifacts/api-worker-smoke-local \
  --algorithm fedprox
```

Với manifest dữ liệu đã audit, chạy FedAvg bằng bốn SuperNode:

```bash
flwr run . local --stream \
  --federation-config \
  "num-supernodes=4 verbose=true backend='ray' client-resources-num-cpus=1 init-args-num-cpus=4" \
  --run-config \
  "algorithm='fedavg' num-server-rounds=30 local-epochs=1 batch-size=32 learning-rate=0.001 seed=2026 client-data-root='data/flower-profiles/dirichlet-alpha-0.5/clients' central-test-manifest='data/flower-profiles/dirichlet-alpha-0.5/test_manifest.csv' output-dir='artifacts/flower/fedavg-alpha-0.5-seed-2026'"
```

Để chạy FedProx:

```bash
flwr run . local --stream \
  --federation-config \
  "num-supernodes=4 verbose=true backend='ray' client-resources-num-cpus=1 init-args-num-cpus=4" \
  --run-config \
  "algorithm='fedprox' proximal-mu=0.01 num-server-rounds=30 local-epochs=1 batch-size=32 learning-rate=0.001 seed=2026 client-data-root='data/flower-profiles/dirichlet-alpha-0.5/clients' central-test-manifest='data/flower-profiles/dirichlet-alpha-0.5/test_manifest.csv' output-dir='artifacts/flower/fedprox-alpha-0.5-seed-2026'"
```

### 7. Chạy dashboard

Tạo credential cục bộ một lần. Lệnh từ chối ghi đè `.env` hoặc khóa đã tồn tại và
không in password/token/private key ra terminal:

```powershell
.\.venv-flower\Scripts\python.exe scripts\generate_local_secrets.py
docker compose up -d --build db api web
docker compose ps
```

Mở `https://localhost:8443`; `http://localhost:8080` tự chuyển sang HTTPS. Chứng
thư local do CA tại `secrets/web-tls/ca.crt` ký, vì vậy trình duyệt sẽ cảnh báo cho
đến khi CA này được tin cậy trên máy demo. Đăng nhập bằng một trong hai token trong
`.env`; không chụp màn hình, commit hoặc gửi các giá trị đó qua chat/email.

Mặc định dashboard chỉ cho chạy synthetic smoke. Muốn xếp hàng Flower, đặt
`CROPFED_FLOWER_WORKER_ENABLED=true`, chuẩn bị ba profile dữ liệu rồi chạy:

```bash
docker compose --profile flower up --build
```

Ngày 30/07/2026, stack `db/api/web` đã được build và chạy thật trên Docker Desktop:
ba healthcheck đều `healthy`, HTTP→HTTPS trả `308`, HTTPS trả security headers,
request thiếu token trả `401`, viewer bị chặn ghi với `403`. Flower worker profile,
TLS/node authentication của Flower và triển khai nhiều máy vẫn chưa được coi là đã
kiểm chứng.

## Thứ tự đọc tài liệu khi mở phiên làm việc mới

1. [00_PROJECT_CONTEXT.md](docs/00_PROJECT_CONTEXT.md) — nguồn sự thật về đề tài và quyết định đã khóa.
2. [DECISION_LOG.md](docs/DECISION_LOG.md) — lý do của từng quyết định.
3. [08_TRACEABILITY_MATRIX.md](docs/08_TRACEABILITY_MATRIX.md) — phần nào đã code, phần nào chưa.
4. [06_ROADMAP_AND_HANDOFF.md](docs/06_ROADMAP_AND_HANDOFF.md) — việc tiếp theo theo đúng thứ tự.

Các tài liệu còn lại:

- [01_SCOPE_AND_REQUIREMENTS.md](docs/01_SCOPE_AND_REQUIREMENTS.md)
- [02_ARCHITECTURE.md](docs/02_ARCHITECTURE.md)
- [03_DATA_AND_NON_IID.md](docs/03_DATA_AND_NON_IID.md)
- [04_EXPERIMENT_PROTOCOL.md](docs/04_EXPERIMENT_PROTOCOL.md)
- [11_PLANTVILLAGE_PILOT_REPORT.md](docs/11_PLANTVILLAGE_PILOT_REPORT.md)
- [05_SECURITY_AND_PRIVACY.md](docs/05_SECURITY_AND_PRIVACY.md)
- [07_REFERENCES.md](docs/07_REFERENCES.md)
- [09_REPORT_OUTLINE.md](docs/09_REPORT_OUTLINE.md)
- [10_TEST_REPORT.md](docs/10_TEST_REPORT.md)

## Cấu trúc chính

```text
src/cropfed/
├── api/          # FastAPI, SQLModel và experiment metadata
├── data/         # manifest, split, IID và Dirichlet Non-IID
├── experiments/  # centralized baseline
├── fl/           # aggregation độc lập framework
├── flower/       # ClientApp và ServerApp
├── ml/           # MobileNetV2/ResNet18, train, evaluate, metric
├── cli.py
└── simulation.py # smoke test tổng hợp, không phải kết quả nghiên cứu
```

## Các ranh giới bắt buộc

- Không đổi tên đề tài.
- Không gọi bài toán hiện tại là object detection.
- Không báo cáo accuracy đơn lẻ; phải có macro F1 và per-class recall/F1.
- Không dùng global test để chọn siêu tham số.
- Không đưa ảnh PlantVillage hoặc ảnh cơ sở nông nghiệp vào repository/bản bàn giao.
- Không tuyên bố FL tự động bảo đảm riêng tư; model update vẫn có thể rò rỉ thông tin.

Nguồn kỹ thuật được xác minh gần nhất ngày 30/07/2026: [Flower 1.32 architecture](https://flower.ai/docs/framework/explanation-flower-architecture.html), [Flower PyTorch quickstart](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html), [Torchvision MobileNetV2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html).
