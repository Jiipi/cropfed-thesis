# 11 — Báo cáo dữ liệu và pilot PlantVillage

Ngày kiểm chứng: 30–31/07/2026. Đây là dữ liệu PlantVillage công khai, không phải
fixture tổng hợp; tuy nhiên pilot một epoch vẫn bị khóa
`research_result_valid=false` và không được dùng làm kết quả nghiên cứu chính.

## 1. Nguồn và phiên bản dữ liệu

- Nguồn: `https://github.com/spMohanty/PlantVillage-Dataset`.
- Commit đã checkout: `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`.
- Citation đi kèm nguồn: Mohanty, Hughes và Salathé (2016), *Using deep
  learning for image-based plant disease detection*, DOI
  `10.3389/fpls.2016.01419`.
- Phạm vi dùng trong đề tài: 18.160 ảnh màu cà chua thuộc đúng 10 lớp đã khóa.
- Raw image nằm dưới `data/raw/` và bị `.gitignore`; source code/bản bàn giao không
  chứa ảnh.

| Lớp | Số ảnh |
|---|---:|
| Tomato healthy | 1.591 |
| Bacterial spot | 2.127 |
| Early blight | 1.000 |
| Late blight | 1.909 |
| Leaf mold | 952 |
| Septoria leaf spot | 1.771 |
| Target spot | 1.404 |
| Tomato mosaic virus | 373 |
| Tomato yellow leaf curl virus | 5.357 |
| Two-spotted spider mite | 1.676 |

## 2. Split, profile và data integrity

Lệnh `prepare-mvp-profiles` đã tạo một split dùng chung gồm 14.529 ảnh train và
3.631 ảnh global test, seed 2026. Ba profile IID, Dirichlet `α=0.5` và Dirichlet
`α=0.1` cùng dùng chính xác hai manifest này.

- Train manifest SHA-256:
  `35dd006bd4fc14a11fcc1049931f37b3f43fdbb37aaa089ec89da74772b07af1`.
- Test manifest SHA-256:
  `fcedfa2b3d93e1aa5dcb5461851d69b786e2009c2430c84d3d37be39c94fed81`.
- Cả ba audit: `passed`, 0 error, hash artifact khớp `profiles_index.json`.
- Có 10 duplicate-content entry trong train và 4 trong test; split theo nhóm
  SHA-256 giữ mỗi nhóm ở cùng một phía nên không có content overlap train/test.
- Không có nhóm nội dung trùng nhưng khác label.

## 3. Centralized MobileNetV2 pilot

Artifact:
`artifacts/plantvillage-centralized-pilot-seed2026/`.

| Thuộc tính | Giá trị |
|---|---:|
| Pretrained ImageNet | Có |
| Epoch | 1 |
| Batch size | 64 |
| Learning rate | 0,001 |
| Seed | 2026 |
| Runtime | 3.186,90 giây |
| Accuracy | 0,9325 |
| Macro precision | 0,9388 |
| Macro recall | 0,8571 |
| Macro F1 | 0,8688 |
| Harmful → healthy rate | 0,00423 |
| Checkpoint bytes | 9.182.987 |

Checkpoint SHA-256:
`6d8124f26936a52d370af8134f125e48f3e66028399814c08d548336912d29b4`.
Loader đã xác nhận format version 1, MobileNetV2 và đúng thứ tự 10 lớp.

Pilot cho thấy chênh lệch lớp rõ rệt: Tomato mosaic virus có recall 0,2267 và
Target spot 0,7153, trong khi nhiều lớp lớn vượt 0,90. Vì chỉ có một epoch và
batch 64 thay cho protocol chính batch 32, kết quả này chỉ là kiểm tra tài nguyên.
Global test đã được dùng để mô tả pilot nên các số này chỉ giúp phát hiện rủi ro;
không được dùng để chọn hyperparameter hoặc kết luận giả thuyết.

## 4. Inference cục bộ

CLI đã nạp checkpoint trên và dự đoán một ảnh từ global test mà không upload ảnh.
Mẫu ground-truth `Tomato healthy` được dự đoán đúng với confidence 0,99969; output
có crop, class/group, top-3, model version và cảnh báo giới hạn. Lần kiểm tra này
đồng thời phát hiện và sửa lỗi `UnicodeEncodeError` khi Windows CP1258 in cảnh báo
tiếng Việt; CLI hiện ép UTF-8 và có regression test riêng.

## 5. Local-only MobileNetV2 pilot — Dirichlet α=0.5

Artifact đạt:
`artifacts/plantvillage-local-only-alpha05-pilot-seed2026-run2/`.
Lần gọi đầu dừng trước training vì Torch cố ghi cache pretrained ngoài workspace;
log được giữ dưới tên `cache-permission-failed`. Run 2 dùng cùng weight đã cache tại
`artifacts/torch-cache`, hoàn tất 4/4 client, stderr rỗng.

Cấu hình giữ seed/model/epoch/batch/learning rate giống centralized pilot. Mỗi
client train trên manifest riêng, đánh giá trên validation cục bộ và cùng global
test 3.631 ảnh.

| Client | Train | Local validation Macro-F1 | Global accuracy | Global Macro-F1 | Harmful → healthy |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.171 | 0,5459 | 0,6447 | 0,4653 | 0,1678 |
| 1 | 2.526 | 0,6775 | 0,7279 | 0,6127 | 0,0293 |
| 2 | 2.159 | 0,7283 | 0,8257 | 0,6784 | 0,0100 |
| 3 | 4.767 | 0,7740 | 0,8491 | 0,7689 | 0,0389 |

- Mean global accuracy: 0,7618.
- Mean global Macro-F1: 0,6313.
- Worst-client global Macro-F1: 0,4653.
- Best-client global Macro-F1: 0,7689.
- Tổng runtime: 1.463,70 giây.
- Bốn checkpoint và environment SHA-256 đều khớp `result.json`;
  `research_result_valid=false` và `pilot_not_for_research`.

Centralized pilot Macro-F1 0,8688 cao hơn cả local-only tốt nhất 0,7689 và trung
bình 0,6313. Đây là tín hiệu pilot phù hợp giả thuyết hợp tác, không phải kết luận
RQ4 vì mới một seed/một epoch.

## 6. Flower FedAvg IID pilot — 1 round, 4 client

Artifact đạt:
`artifacts/plantvillage-flower-fedavg-iid-pilot-seed2026/`.
Run dùng MobileNetV2 pretrained, seed 2026, một local epoch, batch 64, learning
rate 0,001 và profile IID. Bốn client có cùng 2.906 ảnh train; validation cục bộ
lần lượt có 727/726/726/726 ảnh. Global test vẫn là 3.631 ảnh đã khóa ở mục 2.

| Metric sau round 1 | Giá trị |
|---|---:|
| Central accuracy | 0,93335 |
| Central macro precision | 0,93289 |
| Central macro recall | 0,90910 |
| Central macro F1 | 0,91453 |
| Central harmful → healthy rate | 0,01268 (42 ảnh) |
| Central disease F1 | 0,98798 |
| Central pest/spider-mite F1 | 0,85329 |
| Aggregated client validation accuracy | 0,93563 |
| Aggregated client validation macro F1 | 0,90664 |
| Aggregated train loss | 0,61594 |
| Tổng payload gửi + nhận | 109.596.872 byte |
| Strategy runtime | 1.533,55 giây |

Flower nhận đủ 4/4 train reply và 4/4 evaluate reply, không có failure. Checkpoint
SHA-256 là
`46f2d1e2b114d436f6f563bcdf3d52f7ae2f32414d9a9fbd34dd2f4b39ef0ce7`;
hash checkpoint và `environment.json` đều khớp `run_manifest.json`. Server không
nhận ảnh gốc (`raw_images_received_by_server=false`). Artifact được khóa
`result_kind=federated_image_pilot`, `research_result_valid=false` và
`pilot_not_for_research`.

CLI cũng đã nạp trực tiếp `global_model.pt` và dự đoán mẫu đầu tiên của global
test là `Tomato healthy` với confidence 0,99986; output xác nhận
`image_uploaded=false`.

Lần thử đầu dùng bốn Ray actor song song trên Windows bị `ActorDiedError` và chỉ
nhận 0/4 reply; log được giữ tại
`artifacts/run-logs/flower-fedavg-iid-pilot-parallel-actors-failed-20260731.*.log`.
Tracking strategy đã được sửa để dừng ngay nếu số valid reply ít hơn số node đã
gửi, thay vì tiếp tục aggregate/evaluate một kết quả thiếu. Lần đạt dùng một actor
với 8 CPU để xử lý bốn client tuần tự. Ray vẫn in trace `access violation` từ các
worker phụ lúc khởi tạo/thu hồi trên Windows, nhưng actor chính hoàn tất, server
nhận đủ reply, process trả mã 0 và artifact/hash đầy đủ. Khi chạy lại trên Windows
nên giữ cấu hình một actor; deployment chính thức nên ưu tiên Linux.

Exporter thực tế tại
`artifacts/export-plantvillage-pilots-exclusion-with-flower-20260731/` nhận cả ba
artifact centralized/local-only/Flower và trả `included=0`, `excluded=3`; lý do
loại từng run được ghi trong `export_manifest.json`.

Không so sánh 0,9145 của Flower với 0,8688 centralized như kết luận RQ5: đây là
pilot một seed/một epoch, hai đường chạy có cách chia train/validation khác nhau và
global test đã được xem nhiều lần trong quá trình kiểm chứng. Số liệu chỉ chứng
minh pipeline FL thật có thể hoàn tất và tạo artifact truy vết được.

## 7. Flower FedAvg/FedProx Non-IID pilot — α=0.5 và α=0.1

Bốn pilot bổ sung đã chạy trên cùng seed 2026, một round, một local epoch, batch
32, learning rate 0,001 và MobileNetV2 pretrained=false (theo profile PlantVillage
đã khóa). Cấu hình Ray simulation: `client-resources-num-cpus=1` và
`init-args-num-cpus=8`, một actor xử lý bốn client tuần tự (workaround Windows đã
ghi ở mục 6). Mỗi pilot dùng `proximal-mu=0.01` cho FedProx; FedAvg lấy `μ=0.0`.

Tất cả bốn artifact đều đạt verification `passed` theo
`scripts/verify_plantvillage_pilots.py`: SHA-256 checkpoint khớp
`run_manifest.json`, `client_history.json` bám đúng `(round, client_id, phase)` và
`flower.log` có evidence `aggregate_train`/`aggregate_evaluate` của 4/4 reply.

| Pilot | Artifact | Checkpoint SHA-256 (12 ký tự đầu) | Communication bytes | Strategy runtime (s) |
|---|---|---:|---:|---:|
| FedAvg α=0.5 | `artifacts/plantvillage-flower-fedavg-alpha05-pilot-seed2026/` | `642dde796829` | 109.596.872 | ~1.187 |
| FedAvg α=0.1 | `artifacts/plantvillage-flower-fedavg-alpha01-pilot-seed2026/` | `4c5b2ca7bbd8` | 109.596.872 | ~1.153 |
| FedProx α=0.5 | `artifacts/plantvillage-flower-fedprox-alpha05-pilot-seed2026/` | `be9cee850a24` | 109.596.872 | ~1.301 |
| FedProx α=0.1 | `artifacts/plantvillage-flower-fedprox-alpha01-pilot-seed2026/` | `860247cf090a` | 109.596.872 | ~1.115 |

Checkpoint đều ở format version 1, model version `0.1.0`, đúng 10 lớp cà chua và
`raw_images_received_by_server=false`. Bốn pilot đều khóa
`research_result_valid=false`, `protocol_lock=null` và
`result_kind=federated_image_pilot` — phù hợp vai pilot, không dùng cho RQ1–RQ6.

Metric cuối cùng từ `metrics.json` (`global_test_macro_f1`) vẫn thấp (~0,046) do
chỉ một round và một local epoch trên tập Non-IID nặng; đây là kết quả pilot, không
phải so sánh thuật toán. Để khóa `μ` cho main study, cần chạy thêm rounds/local
epochs và đánh giá `μ ∈ {0.001, 0.01, 0.1}` chỉ trên tập validation (chưa thực
hiện tại đây).

Lần chạy đầu của FedAvg α=0.5 gặp `ActorDiedError` trên một worker phụ của Ray
Windows tương tự mục 6, nhận 3/4 reply và tracking dừng sớm; artifact đã được tái
tạo bằng cách chạy lại cùng cấu hình (`status=passed` sau verify).

## 8. Worker profile end-to-end trên Docker Compose

Worker profile trên Compose (`worker` với `profiles: ["flower"]`) đã được verify
qua Docker ngày 31/07/2026. Sau khi sửa `pyproject.toml` thêm
`python-multipart>=0.0.20,<1` (thiếu để FastAPI 0.141 boot) và sửa manifest paths
từ `F:\project\…` thành POSIX qua `scripts/rewrite_profile_paths.py`, worker claim
được experiment từ PostgreSQL, chạy data audit POSIX (`pre_run_data_audit.json` đủ
5 manifests) và spawn Flower subprocess với Ray 8 CPU. Audit fail với 14.529
`FileNotFoundError` do image_id trùng giữa train/val/test nên audit gắn cờ
`client_metadata_mismatch`; đây là vấn đề data prep chứ không phải Docker. Một
round MobileNetV2 CPU với 14.529 ảnh/client vượt 30 phút; cần tăng tài nguyên
hoặc giảm dataset khi chạy full main matrix trong container.

## 9. Việc tiếp theo

1. Pilot FedAvg/FedProx `α=0.5` và `α=0.1`, chọn `μ` chỉ trên validation.
2. Khóa số epoch/round, batch size và tiêu chí early stopping trước main study.
3. Sửa data prep để PlantVillage client manifests chỉ chứa ảnh không trùng
   train/val/test; sau đó chạy Flower main matrix qua Docker worker profile.
4. Chạy seed nghiên cứu chính 2026/2027/2028 mà không dùng global test để tinh chỉnh.
