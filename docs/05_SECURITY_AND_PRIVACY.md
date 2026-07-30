# 05 — Bảo mật và quyền riêng tư

## 1. Tuyên bố chính xác

Federated Learning giúp huấn luyện mà không cần tập trung ảnh thô, nhưng **không tự động bảo đảm riêng tư**. Model update và model cuối vẫn có thể tiết lộ thông tin. NIST tổng hợp các tấn công khôi phục/suy luận từ update và model trong [Privacy Attacks in Federated Learning, 2024](https://www.nist.gov/blogs/cybersecurity-insights/privacy-attacks-federated-learning).

Trong báo cáo nên dùng:

> “Hệ thống giảm việc truyền và lưu trữ tập trung dữ liệu ảnh thô; phạm vi MVP chưa cung cấp bảo đảm riêng tư hình thức như differential privacy.”

Không dùng:

> “Dữ liệu hoàn toàn an toàn/ẩn danh vì dùng FL.”

## 2. Tài sản cần bảo vệ

- ảnh và metadata tại cơ sở;
- model update;
- global checkpoint;
- thông tin phân phối bệnh/số mẫu của từng cơ sở;
- credential/TLS key;
- log và đường dẫn máy;
- tính toàn vẹn của model trước client độc hại.

## 3. Threat model MVP

| Tác nhân | Nguy cơ |
|---|---|
| Người nghe lén mạng | lấy model/update, sửa message |
| Server honest-but-curious | suy luận dữ liệu từ update |
| Client độc hại | model poisoning, gửi NaN/update cực lớn |
| Client bị chiếm quyền | lộ ảnh/path/checkpoint |
| Người dùng web không hợp lệ | chạy job, xem kết quả hoặc làm cạn tài nguyên |
| Lỗi vận hành | commit secret/raw data, log path nhạy cảm |

MVP không tuyên bố chống Byzantine đầy đủ.

## 4. Kiểm soát bắt buộc

### Dữ liệu

- raw images ở client;
- server API không có upload training image;
- `.gitignore` loại data/artifact nhạy cảm;
- log không in full local path nếu triển khai thật;
- retention và backup được cấu hình.

### Giao tiếp/deployment

- TLS giữa SuperNode và SuperLink;
- xác thực node;
- credential qua environment/secret store;
- không dùng password mặc định ngoài local;
- chỉ mở port cần thiết;
- dependency pin/range và scan image.

### Update validation

- parameter key/shape giống global model;
- từ chối NaN/Inf;
- `num_examples > 0`;
- giới hạn message/update size;
- timeout và trạng thái failure;
- nâng cao: norm clipping, robust aggregation, anomaly alert.

### Web

- explicit CORS origins;
- validate range config;
- lifecycle state machine;
- bearer authentication với token `viewer`/`admin` do server cấp;
- token frontend chỉ nằm trong `sessionStorage`, không bundle vào image;
- rate limit, rotation/revocation và secret manager trước public deployment;
- image training chạy worker riêng, không chạy trong request.

### Trạng thái bảo vệ của MVP hiện tại

Đã triển khai cho luồng demo nội bộ:

- Flower worker mặc định tắt và chỉ bật bằng cấu hình phía server;
- HTTP chỉ nhận enum/range đã whitelist, từ chối field lạ;
- dataset profile và output path do server sở hữu;
- subprocess dùng argv cố định, không qua shell;
- data audit bắt buộc trước mỗi Flower job.
- Flower server từ chối update sai schema, NaN/Inf, `num_examples <= 0`, identity
  trùng/ngoài federation và metadata payload vượt 1 MB trước aggregation;
- endpoint phân bố dữ liệu chỉ trả count/proportion từ summary, không trả image byte
  hoặc local path;
- mỗi Flower run mới có environment/checkpoint checksum và không ghi đè artifact cũ.
- Compose local yêu cầu password PostgreSQL cùng hai bearer token khác nhau, dài tối
  thiểu 32 ký tự; generator tạo ngẫu nhiên và từ chối ghi đè secret đã tồn tại;
- API phân quyền `viewer` chỉ đọc và `admin` được ghi; kiểm tra Compose thật xác nhận
  lần lượt `401` khi thiếu token và `403` khi viewer gọi endpoint ghi;
- Nginx local kết thúc TLS 1.2/1.3, chuyển HTTP sang HTTPS và trả HSTS, CSP,
  `X-Content-Type-Options`, `X-Frame-Options` cùng `Referrer-Policy`;
- PostgreSQL và API không publish port ra host; chỉ web publish `8080/8443`.

Các bảo vệ trên mới được kiểm chứng cho demo một máy bằng chứng thư CA cục bộ.
Generator đã tạo material CA/key cho Flower nhưng runtime SuperNode/SuperLink chưa
được cấu hình TLS/node authentication và chưa có rotation/revocation, rate limiting
hay secret manager. Vì vậy không expose worker hiện tại ra Internet hoặc mạng không
tin cậy; các mục này vẫn là điều kiện bắt buộc trước deployment nhiều máy.

## 5. Secure Aggregation và Differential Privacy

### Secure Aggregation

Server chỉ thấy tổng update, không thấy update riêng từng client. Phù hợp nâng cao nếu triển khai nhiều cơ sở thật, nhưng tăng phức tạp khi client dropout.

### Differential Privacy

Clipping + noise giúp giới hạn ảnh hưởng của một record/client và có thể báo privacy budget \((\epsilon,\delta)\). Đổi lại model utility giảm và cần kế toán privacy đúng.

Không thêm DP chỉ để “có tính năng”; nếu làm phải:

- nêu adjacency là sample-level hay client-level;
- nêu clipping norm;
- noise multiplier;
- epsilon, delta và số round;
- so sánh utility/privacy.

## 6. Risk register

| Mã | Rủi ro | Xác suất | Tác động | Giảm thiểu |
|---|---|---|---|---|
| R-01 | PlantVillage không đại diện thực địa | cao | cao | nêu giới hạn; test PlantWild nếu còn thời gian |
| R-02 | Non-IID mạnh làm client quá ít mẫu | vừa | cao | min-size, retry, report distribution |
| R-03 | GPU không đủ/huấn luyện quá lâu | vừa | cao | MobileNetV2, pilot, batch nhỏ, checkpoint |
| R-04 | Flower API thay đổi | vừa | vừa | pin `>=1.32,<2`, theo Message API, integration test |
| R-05 | FedProx cài sai loss | vừa | cao | squared L2, unit/integration test |
| R-06 | Test leakage | vừa | rất cao | split test trước partition, hash intersection |
| R-07 | Accuracy che lớp hiếm | cao | cao | macro/per-class/worst-client metrics |
| R-08 | Update làm rò rỉ dữ liệu | vừa | cao | giới hạn claim; TLS; secure aggregation/DP nâng cao |
| R-09 | Client độc hại phá model | thấp trong mô phỏng | cao | validation, clipping; robust aggregation nâng cao |
| R-10 | Web worker bị treo bởi training | cao nếu thiết kế sai | vừa | job runner riêng |
| R-11 | Kết quả không tái lập | vừa | cao | seed/config/environment/checksum |
| R-12 | Scope creep | cao | cao | khóa MVP; advanced chỉ sau Definition of Done |

## 7. Checklist trước deployment nhiều máy

- [ ] Thay toàn bộ default secret.
- [x] Web TLS local và API authentication/authorization đã bật trong Compose.
- [ ] Flower TLS và node authentication đã nối vào SuperLink/SuperNode.
- [x] Database và API không publish port trong Compose local.
- [ ] Token rotation/revocation, rate limiting và secret manager đã cấu hình.
- [ ] Raw data volume chỉ mount tại đúng client.
- [ ] Log được redaction.
- [ ] Backup/restore đã thử.
- [ ] Update validation và timeout bật.
- [ ] Dependency/image scan không có lỗi nghiêm trọng chưa xử lý.
- [ ] Có incident procedure và cách thu hồi credential.
