# 03 — Dữ liệu và mô phỏng Non-IID

## 1. Dataset chính

PlantVillage là nguồn chính vì:

- có nhãn crop–disease rõ;
- đủ lớn để tạo bốn client mô phỏng;
- được dùng rộng rãi và được dùng trong nghiên cứu FL sâu bệnh gần đề tài;
- nhóm cà chua có cả trạng thái khỏe, nhiều bệnh và lớp nhện hai chấm.

Bài báo gốc báo cáo 54.306 ảnh, 14 loài cây, 38 lớp crop–disease và 26 bệnh hoặc trạng thái khỏe. Xem [Mohanty et al., 2016](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full) và [repository PlantVillage](https://github.com/spmohanty/plantvillage-dataset).

Con số trên mô tả **toàn bộ dataset**, không phải số ảnh của subset cà chua. Số ảnh thực tế của subset phải được lấy từ `partition_summary.json` sau khi scan, không chép một con số không kiểm chứng vào báo cáo.

## 2. Mười thư mục đầu vào

| Folder PlantVillage | Nhãn chuẩn |
|---|---|
| `Tomato___healthy` | Tomato healthy |
| `Tomato___Bacterial_spot` | Bacterial spot |
| `Tomato___Early_blight` | Early blight |
| `Tomato___Late_blight` | Late blight |
| `Tomato___Leaf_Mold` | Leaf mold |
| `Tomato___Septoria_leaf_spot` | Septoria leaf spot |
| `Tomato___Target_Spot` | Target spot |
| `Tomato___Tomato_mosaic_virus` | Tomato mosaic virus |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato yellow leaf curl virus |
| `Tomato___Spider_mites Two-spotted_spider_mite` | Two-spotted spider mite |

CLI từ chối chạy nếu thiếu một folder. Không tự động bỏ qua lớp thiếu vì sẽ làm đổi taxonomy.

## 3. Manifest

Mỗi CSV có đúng các cột:

| Cột | Ý nghĩa |
|---|---|
| `image_id` | SHA-1 rút gọn của relative path, dùng kiểm tra trùng |
| `path` | đường dẫn ảnh tại máy đang giữ dữ liệu |
| `label_id` | số 0–9 theo class contract |
| `label_name` | tên đọc được |
| `split` | `train`, `test`, `local_train` hoặc `local_val` |

Manifest không chứa byte ảnh. Vì path hiện là tuyệt đối, khi đổi máy phải sinh lại manifest cùng seed.

## 4. Thứ tự split bắt buộc

```mermaid
flowchart TB
    ALL["Toàn bộ subset cà chua"] --> HOLD["Global test 20% theo từng lớp"]
    ALL --> POOL["Development pool 80%"]
    POOL --> PART["IID hoặc Dirichlet → 4 client"]
    PART --> TRAIN["Local train 80% mỗi client"]
    PART --> VAL["Local validation 20% mỗi client"]
```

Theo tỷ lệ tổng thể gần đúng:

- local training: 64%;
- local validation: 16%;
- global test: 20%.

Điểm quan trọng là **global test được tách trước khi partition**, nên một ảnh test không thể đi vào bất kỳ client train nào.

## 5. Mô phỏng Non-IID

### IID

Trộn toàn bộ development pool theo seed, chia gần đều bốn client. Đây là đường chuẩn để tách ảnh hưởng của FL khỏi ảnh hưởng của heterogeneity.

### Dirichlet label skew

Với mỗi lớp \(c\), lấy tỷ lệ phân phối qua \(K=4\) client:

\[
p^{(c)} \sim Dirichlet(\alpha \mathbf{1}_K)
\]

Sau đó phân số mẫu của lớp \(c\) theo \(p^{(c)}\). Alpha nhỏ tạo tỷ lệ cực đoan hơn:

- `α=0.5`: Non-IID vừa;
- `α=0.1`: Non-IID mạnh.

Đây là **label distribution skew**, không mô phỏng đầy đủ khác biệt camera, nền, ánh sáng hay địa lý. Tài liệu Flower cũng mô tả alpha nhỏ hơn tạo mức không đồng nhất cao hơn: [DirichletPartitioner](https://flower.ai/docs/datasets/tutorial-use-partitioners.html).

## 6. Điều kiện hợp lệ của partition

Mỗi lần tạo partition phải kiểm tra:

1. tổng index đúng bằng số mẫu nguồn;
2. không index nào xuất hiện hai lần;
3. không mất index;
4. mỗi client có ít nhất hai mẫu trước local split;
5. cùng seed cho cùng kết quả;
6. `partition_summary.json` lưu count và proportion của 10 lớp;
7. vẽ heatmap phân phối lớp trước khi train.

Nếu `α=0.1` không thể thỏa min-size, được phép retry deterministic trong cùng thuật toán; không được chuyển alpha mà không ghi config.

## 7. Data quality checklist

- Mở ngẫu nhiên ảnh từng lớp để phát hiện folder nhầm.
- Đọc thử tất cả ảnh, ghi danh sách file hỏng.
- Kiểm tra RGB conversion.
- Kiểm tra duplicate bằng hash nội dung nếu có nguy cơ cùng ảnh ở nhiều folder.
- Kiểm tra class imbalance.
- Kiểm tra train/val/test intersection bằng `image_id` và content hash.
- Lưu checksum của manifest.
- Không tăng cường dữ liệu ở validation/test.
- Train transform: crop, flip, rotation nhẹ, color jitter nhẹ.
- Eval transform: resize/center crop/normalize cố định.

CLI `cropfed audit-data` hiện thực cổng kiểm tra tự động cho giải mã ảnh,
taxonomy, checksum manifest, content SHA-256, overlap train/test và tính đầy đủ
của client partition. Duplicate chỉ nằm trong cùng train hoặc cùng test được
ghi warning để nghiên cứu viên quyết định; duplicate nội dung cắt qua train/test
là lỗi chặn thí nghiệm.

## 8. Hạn chế của PlantVillage

Ảnh PlantVillage phần lớn có điều kiện chụp được kiểm soát và nền đơn giản. Hiệu năng cao trên bộ này không chứng minh mô hình hoạt động tốt ngoài đồng ruộng. Báo cáo phải:

- gọi đây là benchmark/lab-style dataset;
- không suy rộng thành hệ thống chẩn đoán thực địa hoàn chỉnh;
- đề xuất đánh giá domain shift bằng dữ liệu in-the-wild.

Nguồn nâng cao phù hợp:

- [PlantWild, ACM MM 2024](https://arxiv.org/html/2408.03120v1): ảnh thực địa, nền/góc/ánh sáng đa dạng; dùng để kiểm tra generalization nếu ánh xạ được lớp.
- [IP102, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_IP102_A_Large-Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.html): bài toán nhận dạng côn trùng hại; không trộn thẳng với taxonomy bệnh lá trong MVP.

## 9. Quyền sử dụng và phân phối

- Không đưa raw images vào Git/repository/bản zip bàn giao.
- Giữ citation và đọc điều khoản của nguồn dataset trước khi chia sẻ.
- PlantWild công bố giấy phép CC BY-NC-ND 4.0 trên trang chính thức; cần tuân thủ nếu dùng.
- Nếu bổ sung ảnh cơ sở thực, cần đồng ý thu thập, chính sách lưu giữ và ẩn metadata vị trí/thiết bị nếu cần.

## 10. Lệnh chuẩn

```bash
cropfed prepare-mvp-profiles \
  --dataset-root /absolute/path/to/plantvillage/color \
  --output-root data/flower-profiles \
  --test-fraction 0.2 \
  --validation-fraction 0.2 \
  --clients 4 \
  --seed 2026
```

Lệnh tạo đúng ba thư mục `iid`, `dirichlet-alpha-0.5` và
`dirichlet-alpha-0.1`. Global train/test được tách đúng một lần rồi ghi byte-identical
vào cả ba profile; `profiles_index.json` xác nhận invariant bằng SHA-256. Mỗi profile
có `partition_summary.json`, `data_audit.json` và `profile.json`. Lệnh từ chối ghi đè
output không rỗng. `prepare-data` và `audit-data` vẫn dùng được khi cần xử lý riêng một
profile.

Chỉ bắt đầu pilot khi report có `status=passed`.
