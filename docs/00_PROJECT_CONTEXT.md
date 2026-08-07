# 00 — Hồ sơ ngữ cảnh bất biến

> Khi mở một phiên làm việc mới, đọc hết tài liệu này trước khi sửa code hoặc đổi thiết kế.

## 1. Tên đề tài chính thức

**“Nghiên cứu và xây dựng hệ thống học liên kết (Federated Learning) cho phát hiện sâu bệnh cây trồng qua ảnh trên dữ liệu phân tán, không đồng nhất giữa các cơ sở nông nghiệp.”**

Không được đổi tên, rút gọn tên trong bìa báo cáo, hoặc chuyển đề tài thành một bài toán khác.

## 2. Cách diễn giải bài toán đã khóa

- “Phát hiện qua ảnh” trong phiên bản đồ án = **phân loại đa lớp ở mức toàn ảnh**.
- Input: một ảnh lá cây trồng.
- Output: xác suất và nhãn của một trong 38 lớp sâu/bệnh/khỏe (xem §6; phạm vi 10
  lớp cà chua là mốc 0.1.0, đã được thay bằng D-037).
- Không có bounding box, mask hay đếm số côn trùng trong MVP.
- Cách diễn giải này phù hợp với bài báo *Image-based crop disease detection with federated learning* (Scientific Reports, 2023): tiêu đề dùng “detection” nhưng thí nghiệm thực hiện crop disease classification.

## 3. Bài toán thực tế

Nhiều cơ sở nông nghiệp có ảnh cây trồng nhưng dữ liệu:

- nằm rải rác ở từng nơi;
- khác nhau về giống, bệnh phổ biến, thiết bị, ánh sáng và số lượng;
- khó hoặc không nên gom toàn bộ ảnh về một máy chủ;
- có phân phối không đồng nhất (Non-IID).

Hệ thống cần học một mô hình chung bằng cách gửi mô hình tới cơ sở, huấn luyện tại chỗ và chỉ trả model update/metric về server. Mục tiêu nghiên cứu không phải chứng minh “FL luôn tốt hơn centralized”, mà đo chênh lệch hiệu năng và chi phí khi dữ liệu ngày càng không đồng nhất.

## 4. Câu hỏi nghiên cứu

| Mã | Câu hỏi |
|---|---|
| RQ1 | Federated Learning có xây dựng được mô hình nhận dạng sâu bệnh từ dữ liệu phân tán mà không truyền ảnh huấn luyện gốc lên server không? |
| RQ2 | Label skew Non-IID ở `α=0.5` và `α=0.1` ảnh hưởng thế nào đến accuracy, macro F1, worst-client F1 và tốc độ hội tụ? |
| RQ3 | FedProx có ổn định hoặc hiệu quả hơn FedAvg khi mức độ Non-IID tăng hay không? |
| RQ4 | Global model có tốt hơn các local-only model của từng cơ sở trên cùng global test set không? |
| RQ5 | Global FL model chênh lệch thế nào so với centralized model khi dùng cùng split, initialization và protocol? |
| RQ6 | Mô hình nhẹ có giảm thời gian, kích thước và dung lượng truyền mà vẫn giữ hiệu quả nhận dạng hợp lý không? |

RQ1–RQ5 là câu hỏi cốt lõi của MVP. RQ6 chỉ được kết luận đầy đủ khi có ít
nhất một backbone đối chiếu (dự kiến ResNet18); nếu chưa đủ tài nguyên thì chỉ
báo các số đo tuyệt đối của MobileNetV2 và ghi rõ giới hạn so sánh.

## 5. Giả thuyết làm việc

- H1: Non-IID càng mạnh thì FedAvg càng dễ dao động hoặc giảm hiệu quả.
- H2: FedProx có thể ổn định hơn FedAvg trong Non-IID mạnh, nhưng cần chọn `μ` trên validation và không giả định luôn thắng.
- H3: Global FL model tốt hơn phần lớn local-only model trên cùng global test set.
- H4: Centralized có thể tốt hơn FL, nhưng không đáp ứng ranh giới không tập trung ảnh gốc.
- H5: MobileNetV2 tạo cân bằng tốt giữa macro F1, thời gian và dung lượng model so với backbone đối chiếu.

Các câu trên là giả thuyết cần kiểm chứng, không phải kết luận viết sẵn.

## 6. Phạm vi dữ liệu và lớp

Dataset chính: **PlantVillage bản đầy đủ, 38 lớp trên 14 loại cây** (D-037,
06/08/2026). Bảng 10 lớp cà chua bên dưới là phạm vi mốc 0.1.0, giữ lại vì mọi
checkpoint và artifact trước 06/08 dùng đúng thứ tự ID đó.

Taxonomy hiện hành nằm ở `constants.py: PLANTVILLAGE_FULL_TAXONOMY`
(`scope="plantvillage-full"`) — **38 lớp: 25 disease, 12 healthy, 1 pest**
(Two-spotted spider mite). 14 cây: Apple, Bell pepper, Blueberry, Cherry, Corn,
Grape, Orange, Peach, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
Trên dữ liệu thật, seed 2026: 54.305 ảnh chia **43.447 development pool / 10.858
global test**; trong pool đó có **34.757 local train + 8.690 local validation**.

Thứ tự lớp là contract, không sắp xếp lại sau khi đã tạo checkpoint. Hai taxonomy
cùng tồn tại là có chủ đích: `taxonomy_from_class_order` nhận diện checkpoint cũ
thuộc taxonomy nào và trả `None` khi không khớp cái nào, thay vì ngầm giả định một
cái — nhầm taxonomy sẽ cho ra nhãn sai mà vẫn chạy trót lọt.

### Phạm vi mốc 0.1.0 — PlantVillage nhóm cà chua, 10 lớp (`scope="tomato"`)

| ID | Nhãn chuẩn trong code | Loại |
|---:|---|---|
| 0 | Tomato healthy | khỏe |
| 1 | Bacterial spot | bệnh |
| 2 | Early blight | bệnh |
| 3 | Late blight | bệnh |
| 4 | Leaf mold | bệnh |
| 5 | Septoria leaf spot | bệnh |
| 6 | Target spot | bệnh |
| 7 | Tomato mosaic virus | bệnh |
| 8 | Tomato yellow leaf curl virus | bệnh |
| 9 | Two-spotted spider mite | sâu/hại |

Thứ tự ID là contract giữa dataset, model, API và báo cáo. Không sắp xếp lại sau khi đã tạo checkpoint.

## 7. Cấu hình chuẩn

- Cross-silo simulation: 4 client đại diện 4 cơ sở.
- Phân phối: **6 profile** — IID; Dirichlet label skew `α=100`, `α=0.5`, `α=0.1`;
  quantity skew; feature skew. `α=100` được thêm theo §6 đề cương để có một mức
  "gần IID nhưng không phải IID": nếu không có nó, bước nhảy từ IID sang `α=0.5`
  là bước duy nhất và không nói được đường suy giảm bắt đầu từ đâu.
- Model chính: MobileNetV2 với transfer learning ImageNet.
- Đối chiếu bắt buộc: local-only, centralized, FedAvg, FedProx. **Mở rộng theo
  D-030**: thêm FedBN, SCAFFOLD, MOON — §2/§5 GĐ3 đề cương yêu cầu, và code đã cài
  5 thuật toán trong khi D-007 mới chốt 4.
- Model phụ nếu còn thời gian: ResNet18.
- Seed phát triển: `2026`.
- Báo cáo chính thức: ưu tiên ba seed `2026, 2027, 2028`; nếu tài nguyên không đủ phải ghi rõ giới hạn.
- Global test: tách trước khi phân dữ liệu cho client.
- Metric chính: macro F1; kèm accuracy, macro precision/recall, per-class recall/F1,
  confusion matrix, worst-client F1, **độ lệch giữa client (std/spread) và khoảng
  cách vs centralized** (D-036 — worst-client F1 là sàn, không phải độ lệch).
- Metric hệ thống: thời gian, round hội tụ, model bytes, tổng upload/download.

## 8. MVP bắt buộc

1. Chuẩn bị manifest và kiểm tra taxonomy (38 lớp; 10 lớp ở mốc 0.1.0).
2. Tạo bốn client IID/Non-IID tái lập bằng seed.
3. Centralized MobileNetV2.
4. Flower FedAvg.
5. Flower FedProx.
6. Lưu metric theo round và checkpoint.
7. Dashboard xem cấu hình/trạng thái/kết quả.
8. Inference một ảnh tại client.
9. So sánh có kiểm soát giữa centralized, local-only và FL.
10. Báo cáo giới hạn riêng tư; không tuyên bố FL là privacy guarantee.

## 9. Nâng cao, chỉ làm sau MVP

- ~~SCAFFOLD hoặc FedAdam/FedYogi~~ — SCAFFOLD **đã vào phạm vi chính** (D-030).
- Secure Aggregation.
- Differential Privacy với báo cáo epsilon/delta. `fl/privacy.py` đã tồn tại nhưng
  **chưa được gọi từ đâu và chưa có test**, nên đây vẫn là "chưa có", không phải
  "đã có" (D-014).
- Client dropout/straggler.
- PlantWild/PlantDoc để đánh giá domain shift.
- Triển khai thật trên nhiều máy.
- Object detection/segmentation.
- Mobile app/edge quantization. `ml/quantization.py` cũng đã tồn tại nhưng chưa nối
  vào CLI/script nào — cùng tình trạng như trên.

## 10. Ngoài phạm vi

- Chẩn đoán y khoa/nông học thay chuyên gia.
- Khuyến nghị thuốc và liều lượng.
- Thu thập dữ liệu thật từ nông dân nếu chưa có quy trình đồng ý.
- Blockchain, IoT sensor, drone video trong MVP. (Multi-crop đã vào phạm vi chính từ
  D-037: 14 loại cây, nhưng vẫn là **một** bài toán phân loại chứ không phải nhiều
  pipeline riêng.)
- So sánh quá nhiều backbone làm loãng câu hỏi Non-IID.

## 11. Ranh giới quyền riêng tư

- Ảnh và manifest chứa đường dẫn ảnh ở client.
- Server nhận: model parameters/update, số mẫu, metric và metadata.
- FL giảm nhu cầu chuyển dữ liệu thô nhưng model update có thể bị tấn công suy luận/khôi phục.
- MVP: TLS khi triển khai, định danh client, log tối thiểu, kiểm tra NaN/shape/update size.
- Secure aggregation và DP là lớp tăng cường, không được ngầm tuyên bố đã có.

## 12. Trạng thái code tại mốc 0.1.0

Đã có:

- partition IID/Dirichlet;
- kiểm tra aggregation FedAvg;
- metric multiclass;
- smoke simulator FedAvg/FedProx;
- PlantVillage manifest, client split và bộ tạo/audit ba profile dùng chung global test;
- dữ liệu PlantVillage tomato thật tại commit nguồn
  `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`: 18.160 ảnh, ba profile audit đạt,
  14.529 train/3.631 global test và không có content overlap qua split;
- data integrity audit cho ảnh hỏng, taxonomy, SHA-256, overlap và client assignment;
- versioned checkpoint có class order/checksum và inference trả crop/group/top-k/model version;
- MobileNetV2/ResNet18, trainer và FedProx loss;
- Flower `ClientApp`/`ServerApp`, round/client metrics, timing, payload bytes, environment artifact và 4-client FedAvg/FedProx integration smoke;
- centralized/local-only runner đã qua smoke ảnh với checkpoint version hóa;
- centralized MobileNetV2 pretrained pilot một epoch trên PlantVillage thật đã hoàn
  tất, checkpoint/hash hợp lệ và được khóa `research_result_valid=false`;
- local-only α=0.5 pilot đủ bốn client đã hoàn tất, mọi checkpoint/hash hợp lệ và
  được khóa `research_result_valid=false`;
- Flower FedAvg IID PlantVillage pilot một round đã hoàn tất đủ 4/4 train và
  evaluate reply, central Macro-F1 0,9145; checkpoint/environment hash hợp lệ và
  được khóa `research_result_valid=false`;
- local single-image inference;
- FastAPI experiment API; worker riêng claim Flower job bằng profile/argv whitelist; DB lưu round và client-phase metric;
- React dashboard cho status, metric nguy cơ, communication, Macro-F1 curve, confusion matrix và heatmap phân bố client;
- exporter comparison/per-class/confusion/environment tự loại synthetic smoke;
- bearer authentication phân quyền viewer/admin, TLS local qua Nginx và secret
  generator không ghi đè;
- Docker Compose `db/api/web` đã build/up thật và cả ba healthcheck đạt;
- 60 test + 2 subtest (gồm Torch, data profile/audit, metric nguy cơ,
  FastAPI/SQLite, auth/secrets/Compose contract, worker và Flower result),
  compile/lint check và frontend build.

Chưa hoàn tất:

- pilot FedAvg/FedProx trên α=0.5/0.1, chốt protocol rồi chạy main matrix nhiều
  seed;
- chạy Flower worker profile trên PostgreSQL và kiểm thử migration/backup/restore;
- inference UI tại client;
- Flower TLS/node authentication, rate limit, secret rotation và deployment nhiều máy;
- ba seed và bảng/biểu đồ báo cáo.

### 12.1 Thêm sau mốc 0.1.0 (06–07/08/2026)

Mục 12 ở trên là ảnh chụp mốc 0.1.0, giữ nguyên. Từ đó tới nay:

- **taxonomy 38 lớp** thay cho 10 lớp cà chua (D-037), 43.447 train / 10.858 test;
- **6 profile** thay cho 3: thêm `dirichlet-alpha-100`, `quantity-skew`,
  `feature-skew`, tất cả dùng chung một global test set (D-024 kiểm lại trên đĩa);
- **5 thuật toán** thay cho 4: FedBN/SCAFFOLD/MOON thực sự chạy đúng thuật toán và
  server raise nếu thiếu trạng thái, thay vì âm thầm chạy như FedAvg (D-030, D-031);
- **launcher main study** chạy được 15 scenario, metadata phân hoạch đọc từ
  `profile.json` chứ không suy từ tên thư mục (D-032);
- **artifact chạy được trên máy khác**: manifest dùng path tương đối theo dataset
  root cấp lúc chạy, thiếu root thì raise; sinh protocol lock cho cả 15 scenario
  (D-033, D-034, D-035);
- **metric fairness và gap**: độ lệch giữa client (std/spread) và khoảng cách vs
  centralized thành cột, thay vì chỉ có worst-client F1 (D-036);
- **230 test + 128 subtest** thay cho 60 + 2.

Hai module đã tồn tại nhưng **chưa nối vào đâu và chưa có test**:
`fl/privacy.py` và `ml/quantization.py`. Chúng nằm ở §9 (nâng cao), không ở §12
(đã có) — có file không đồng nghĩa với có năng lực.

Danh sách khoảng cách còn lại so với đề cương:
[12_PROPOSAL_GAP_CHECKLIST.md](12_PROPOSAL_GAP_CHECKLIST.md).

Chi tiết truy vết: [08_TRACEABILITY_MATRIX.md](08_TRACEABILITY_MATRIX.md).
Kết quả kiểm thử tại mốc bàn giao: [10_TEST_REPORT.md](10_TEST_REPORT.md).
Kết quả dữ liệu/pilot thật: [11_PLANTVILLAGE_PILOT_REPORT.md](11_PLANTVILLAGE_PILOT_REPORT.md).

## 13. Quy tắc chống mất ngữ cảnh

Mọi thay đổi làm ảnh hưởng một mục đã khóa phải:

1. ghi lý do vào [DECISION_LOG.md](DECISION_LOG.md);
2. cập nhật tài liệu liên quan;
3. cập nhật `configs/experiment_matrix.csv` nếu ảnh hưởng thí nghiệm;
4. thêm/chỉnh test;
5. không ghi đè checkpoint hoặc kết quả cũ;
6. ghi version, seed, commit và config trong artifact kết quả.

## 14. Thuật ngữ ngắn

- **Client:** cơ sở nông nghiệp giữ dữ liệu và huấn luyện cục bộ.
- **Server/coordinator:** điều phối round và tổng hợp update.
- **Cross-silo:** số tổ chức ít, tương đối ổn định; phù hợp bốn cơ sở mô phỏng.
- **IID:** các client có phân phối dữ liệu gần giống nhau.
- **Non-IID:** phân phối giữa client khác nhau.
- **Label skew:** tỷ lệ nhãn/bệnh khác nhau giữa client.
- **Client drift:** local model đi theo hướng riêng do dữ liệu cục bộ lệch.
- **FedAvg:** trung bình tham số có trọng số theo số mẫu.
- **FedProx:** FedAvg kèm hạng phạt giữ local model gần global model.
- **Macro F1:** tính F1 từng lớp rồi lấy trung bình đều, không để lớp lớn lấn át.
