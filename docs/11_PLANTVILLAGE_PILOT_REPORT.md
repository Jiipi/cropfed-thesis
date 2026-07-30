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
RQ4 vì mới một seed/một epoch và chưa có global FL model PlantVillage để đối chiếu.

## 6. Việc tiếp theo

1. Chạy Flower FedAvg IID 1 round trên chính manifest/checksum này.
2. Chạy FedAvg/FedProx `α=0.5` và `α=0.1` sau khi pilot tài nguyên đạt.
3. Chỉ sau đó mới khóa số epoch/round, `μ` và chạy seed nghiên cứu chính.
