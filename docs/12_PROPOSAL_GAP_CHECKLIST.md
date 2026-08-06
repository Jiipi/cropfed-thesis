# 12 — Checklist khoảng trống đề cương ↔ code

Ngày đối chiếu: 06/08/2026. Nguồn spec: `DeCuongDoAn_FederatedLearningPhatHienSauBenh.pdf`
và `Nghiên cứu và xây dựng hệ thống Học liên kết…pdf`.

**Hai PDF có nội dung giống hệt nhau** — cùng một bản đề cương 4 trang. Đã extract
cả hai (`tmp/thesis_pdf.txt`, `tmp/thesis_main_pdf.txt`) và so sánh: chỉ có một spec
để đối chiếu, không phải hai tài liệu khác nhau.

Mỗi mục dưới đây ghi rõ **trích dẫn đề cương**, **trạng thái code đã xác minh** và
**việc cần làm**. Không mục nào được chuyển sang `[x]` nếu chưa có test hoặc artifact
chứng minh.

---

## A. Ba loại skew trong mô phỏng non-IID — ĐÃ SỬA 06/08/2026

> §5 GĐ2: "Chia dữ liệu phân tán theo Dirichlet (**lệch nhãn, lệch số lượng, lệch
> đặc trưng**); đo suy giảm của FedAvg so với tập trung"

Đề cương yêu cầu **ba** loại skew. Hàm lõi đã có đủ ba, nhưng hai trong ba **không
có đường nào gọi tới từ pipeline tạo profile** — là code chết.

| Loại skew | Hàm lõi | Có test lõi | Nối vào profile |
|---|---|---|---|
| Lệch nhãn (label) | `dirichlet_partition` `partitioning.py:37` | có | **có** |
| Lệch số lượng (quantity) | `_apply_quantity_skew` `partitioning.py:130` | có | **✓ ĐÃ NỐI** |
| Lệch đặc trưng (feature) | `feature_skew_partition` `partitioning.py:186` | **✓ 4 test mới** | **✓ ĐÃ NỐI** |

Bằng chứng là code chết: `make_partitions` được gọi tại `manifest.py:267` mà **không
truyền** `quantity_skew` lẫn `feature_skew_strength`:

```python
grouped_partitions = make_partitions(
    group_labels, num_clients, partition_kind,
    alpha=alpha, seed=seed, min_size=2,
)   # thiếu quantity_skew=, feature_skew_strength=
```

Chuỗi bị đứt ở 3 tầng: `DataProfileSpec` (`profiles.py:31`) chỉ có 3 trường
`name/partition_kind/dirichlet_alpha` → `write_client_manifests` (`manifest.py:234`)
không có tham số skew → `make_partitions` không nhận được gì.

Hệ quả quan sát được: `FULL_PROFILE_SPECS` (`profiles.py:47`) chỉ có 4 profile toàn
label-skew, và `data/flower-profiles-full/` sinh ra đúng 4 thư mục đó
(`iid`, `dirichlet-alpha-100`, `dirichlet-alpha-0.5`, `dirichlet-alpha-0.1`).
Không có profile quantity/feature skew nào tồn tại.

- [x] A1. Thêm `quantity_skew: bool = False` và `feature_skew_strength: float = 0.5`
      vào `DataProfileSpec` (`profiles.py:32-40`).
- [x] A2. Thêm hai tham số tương ứng vào `write_client_manifests` và truyền xuống
      `make_partitions`.
- [x] A3. `partition_summary.json` ghi thêm `skew_type` (`none`/`label`/`quantity`/
      `feature`), `quantity_skew` và `feature_skew_strength`, nên artifact tự mô tả
      được mình. `api/data_profiles.py` **từ chối** summary mâu thuẫn với spec thay
      vì hiển thị nhầm thành label-skew (`test_api_data_profiles.py:78`).
- [x] A4. `FULL_PROFILE_SPECS` nay có 6 profile, thêm `quantity-skew` và
      `feature-skew` (`profiles.py:51-58`).
- [x] A5. 11 test mới: 4 test lõi partitioner, 5 test profile end-to-end, 5 test
      biên API (xem bảng bằng chứng dưới).
- [x] A6. `api/data_profiles.py` trả `skew_type`/`quantity_skew`/
      `feature_skew_strength`; `_validated_clients` chấp nhận `feature_skew`.
      Dashboard `App.jsx:598` đổi nhãn theo `skew_type` — trước đây quantity và
      feature skew đều hiển thị thành `Non-IID α=null`.

**Ràng buộc D-024 (quan trọng):** `_prepare_empty_output_root` (`profiles.py:243`)
từ chối ghi vào thư mục không rỗng. `data/flower-profiles-full/` đã có 54.305 ảnh
partition xong. Profile skew mới **phải tái dùng đúng train/test manifest hiện có**
(cùng seed 2026), không được sinh split mới — nếu không sẽ phá vỡ D-024 "mọi profile
chia sẻ cùng một global test set" và mọi so sánh giữa các profile mất hiệu lực.

**Cách giữ D-024:** không nới lỏng `_prepare_empty_output_root`. Thay vào đó thêm
một cửa hẹp riêng, `extend_data_profiles` (`profiles.py:185`), **không bao giờ quét
lại dataset và không bao giờ tính split mới**. Nó `shutil.copyfile` nguyên bytes
train/test manifest từ profile nguồn và chỉ chia lại train records cho client, nên
D-024 đúng **do cấu trúc** chứ không do quy ước. Trước khi ghi, nó đối chiếu SHA-256
của manifest nguồn với giá trị trong `profiles_index.json` và từ chối nếu lệch; nó
cũng từ chối khác taxonomy, khác số client, khác seed, và từ chối ghi đè profile đã
tồn tại. Chạy qua `cropfed extend-full-profiles`.

### Lỗi thật phát hiện khi sửa

1. **`feature_skew_partition` hỏng với lớp nhỏ hơn số client.** Nhánh chia khối
   giả định mỗi lớp có ít nhất `num_clients` mẫu; lớp hiếm tạo split rỗng sai lệch.
   Đã thêm nhánh xoay vòng theo `class_position` để không client nào bị bỏ đói ở
   mọi lớp hiếm. Fixture 10 lớp không chạm tới, nhưng taxonomy 38 lớp thì có.
2. **Cả hai skew mới đều bỏ qua `min_size`.** `write_client_manifests` chia **nhóm
   nội dung** chứ không chia ảnh, và một client chỉ còn 1 nhóm thì không tách được
   train/validation cục bộ — `manifest.py:308` ném lỗi. `_apply_quantity_skew` chỉ
   kẹp `sizes >= 1`, còn `feature_skew_partition` không kẹp gì. Nay cả hai nhận
   `min_size`, phân bổ phần dư **trên** sàn, và `_enforce_min_size` chuyển mẫu từ
   client lớn nhất khi cần. Nếu sàn bất khả thi thì báo `ValueError` nêu rõ
   `min_size` thay vì để lỗi khó hiểu nổi lên ở tầng dưới.

### Phạm vi của "lệch đặc trưng" — nói đúng, không nói quá

`feature_skew_partition` là xấp xỉ **ở mức lấy mẫu**: mỗi client nhận một tập ảnh
khác nhau, không giao nhau, **trong cùng một lớp**, nên phân phối đặc trưng mỗi
client khác nhau qua việc site đó giữ những bức ảnh nào. Nó **không sửa pixel**, nên
**không** tái tạo covariate shift thật (camera khác, ánh sáng khác, nền khác); muốn
vậy phải có transform ảnh riêng cho từng client trong data loader. Docstring
(`partitioning.py:192`) ghi đúng giới hạn này để luận văn không tuyên bố quá.

### Bằng chứng

| Hạng mục | Bằng chứng |
|---|---|
| Test lõi partitioner | `tests/unit/test_partitioning.py` — bảo toàn mẫu + tái lập, lớp nhỏ hơn số client, sàn `min_size` (quét 12 seed × 2 loại skew), báo lỗi khi sàn bất khả thi |
| Test profile end-to-end | `tests/unit/test_data_profiles.py` — skew khai báo phải tới được partitioner (không âm thầm rơi về IID), đủ ba loại skew trong `FULL_PROFILE_SPECS` |
| Test D-024 | `test_extension_reuses_the_existing_split_and_keeps_d024` khẳng định manifest **giống nhau từng byte**, không chỉ giống số dòng; cộng 3 test từ chối (ghi đè, sai seed, manifest bị sửa) |
| Test biên API | `tests/unit/test_api_data_profiles.py` — 5 test, gồm D-025 (payload không chứa path/byte ảnh) và từ chối summary mâu thuẫn spec |
| Suite | `pytest tests --ignore=tests/api` → **91 passed, 40 subtests passed**. `tests/api` không chạy được vì thiếu `fastapi` trong venv — có từ trước, không liên quan thay đổi này. |
| Lint | `ruff check` sạch trên toàn bộ file đã sửa |
| Artifact | `cropfed extend-full-profiles --output-root data/flower-profiles-full` → `added=['quantity-skew','feature-skew'], total=6, shared_test=True, shared_train=True`. Cả 6 profile `data_audit.json` = `passed`. Kiểm tra lại **trên đĩa** (không tin index): băm SHA-256 6 file `test_manifest.csv` cho ra **1 giá trị duy nhất** — D-024 giữ được, tương tự `train_manifest.csv`. |

### Số đo trên dữ liệu thật (43.447 train / 10.858 test, 38 lớp, seed 2026)

| Profile | skew_type | Số ảnh mỗi client | Tổng |
|---|---|---|---|
| `iid` | none | 10862 / 10862 / 10861 / 10862 | 43447 |
| `dirichlet-alpha-100` | label¹ | 10692 / 10601 / 10991 / 11163 | 43447 |
| `dirichlet-alpha-0.5` | label¹ | 5764 / 14605 / 13819 / 9259 | 43447 |
| `dirichlet-alpha-0.1` | label¹ | 9579 / 11551 / 10186 / 12131 | 43447 |
| **`quantity-skew`** | quantity | **2044 / 33042 / 2041 / 6320** | 43447 |
| **`feature-skew`** | feature | 10488 / 10492 / 10780 / 11687 | 43447 |

¹ Bốn profile cũ sinh ra **trước** A3 nên `partition_summary.json` của chúng chưa có
khoá `skew_type`; `_read_profile` mặc định về giá trị suy ra từ spec (`data_profiles.py:77`)
nên vẫn đọc `ready` bình thường, không cần partition lại 54.305 ảnh.

Hai điểm đáng chú ý, đúng như thiết kế:

- **Quantity skew lệch 16×** (2.041 → 33.042 ảnh) mà tổng vẫn đúng 43.447 và client
  nhỏ nhất vẫn còn 1.633 train + 408 validation. Đây chính là thứ mà một lần âm thầm
  rơi về IID **không** tạo ra được.
- **Feature skew giữ nhãn cân bằng**: cả 4 client đều có **đủ 38 lớp**, tỷ lệ lớp lớn
  nhất chỉ 0,104–0,110. Khác biệt nằm ở **ảnh nào** mỗi site giữ, không ở nhãn nào —
  đúng định nghĩa để không lẫn với label skew.

Toàn bộ 6 profile đọc `status=ready` qua đúng code path của dashboard
(`_read_profile` + `FULL_PROFILE_SPECS`, 38 lớp).

---

## B. Thuật toán chịu non-IID — ĐÃ SỬA 06/08/2026

> §2: "Nghiên cứu, cải tiến hoặc đề xuất thuật toán tổng hợp chịu được non-IID
> (FedProx, **FedBN, SCAFFOLD, MOON** hoặc biến thể đề xuất)"
> §5 GĐ3: "Triển khai và so sánh FedProx, FedBN, SCAFFOLD, MOON"

Trạng thái sau khi sửa (20 test mới, toàn bộ đạt):

| Tầng | fedavg | fedprox | fedbn | scaffold | moon |
|---|---|---|---|---|---|
| `fl/aggregation.py:275` | ✓ | ✓ | ✓ `:114` | ✓ `:197` | ✓ `:261` |
| `flower/server_app.py:44` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ml/trainer.py:65` | ✓ | ✓ `proximal_mu` | — | ✓ `scaffold_*` | ✓ `moon_*` |
| `flower/client_app.py` | ✓ | ✓ | ✓ (no-op) | **✓ ĐÃ NỐI** | **✓ ĐÃ NỐI** |
| Whitelist API/CLI/config | ✓ | ✓ | ✓ | ✓ | ✓ |
| Test | ✓ | ✓ | **✓ 5 test** | **✓ 9 test** | **✓ 11 test** |

Khoảng trống thực tế **sâu hơn mô tả ban đầu**. Ngoài client không truyền tham số,
còn ba lỗi nữa được phát hiện khi sửa:

1. `TrackedSCAFFOLD` **không hề gửi** control variate `c` xuống client, và
   `TrackedMOON` giữ `moon_temperature`/`moon_mu` nhưng **không bao giờ đưa vào
   ConfigRecord**. Nên B1/B2 không thể sửa chỉ ở phía client.
2. `scaffold-server-lr`, `moon-temperature`, `moon-mu` **thiếu hẳn** trong
   `[tool.flwr.app.config]`; `run_config.get(...)` âm thầm dùng default.
3. `_extract_representation` bọc nhánh ResNet trong `torch.no_grad()` và
   `.detach()` — MOON loss sẽ có **gradient bằng 0** trên `resnet18` ngay cả sau
   khi đã nối dây. Đã bỏ detach và thêm test canh gác.

Ngoài ra `_compute_scaffold_c_i_delta` dùng `K = num_examples * epochs`; đúng
công thức SCAFFOLD phải là số **optimizer step**, nếu không control variate bị
chia nhỏ đi đúng bằng batch size. Đã sửa thành `local_steps` và có test.

- [x] B1. Truyền `scaffold_control_variate` / `scaffold_server_c` từ
      `msg.content` vào `train_local`; gửi delta trả server qua ArrayRecord
      riêng `scaffold_c_delta` (`client_app.py`, `tracking.py`).
- [x] B2. Truyền `moon_previous_model` / `moon_global_model` / `moon_temperature` /
      `moon_mu`; client giữ model round trước trong `Context.state`.
- [x] B3. Test cho `fedbn`/`scaffold`/`moon` — `tests/flower/test_flower_algorithms.py`
      (14 test), `tests/unit/test_trainer_algorithms.py` (6 test) và 5 test FedBN
      mới trong `tests/unit/test_aggregation.py` (BN stats không được average).
- [x] B4. Test khẳng định `scaffold`/`moon` **thực sự khác** FedAvg ở tensor level
      theo tiền lệ D-020 — hai epoch, cùng seed, so sánh từng tensor.
- [x] B5. Fail-fast: server raise nếu client không gửi control variate
      (`tracking.py`) hoặc không báo `moon_contrastive_loss` từ round 2; client
      raise nếu server không gửi `c` / `moon-mu`.
- [x] B6. `DECISION_LOG` D-030 ghi nhận mở rộng vượt D-007 và lý do fail-fast.

**Ràng buộc quan trọng đã xử lý:** Flower gộp *mọi* ArrayRecord trong một reply
theo key. Vì delta SCAFFOLD trùng tên tham số với model, để nguyên sẽ cộng thẳng
control variate vào trọng số. `_take_auxiliary_arrays` phải tách record phụ
**trước** khi aggregation của FedAvg chạy — đã có test canh gác riêng cho đúng
thứ tự này (kiểm chứng bằng cách đảo thứ tự: test fail đúng như mong đợi).

---

## C. Metric fairness và gap-vs-centralized — P1

> §7: "báo cáo **KHOẢNG CÁCH** so với mô hình tập trung"
> §7: "**Tính công bằng** — độ lệch accuracy giữa cơ sở mạnh và yếu; hệ phân tán tốt
> không được bỏ rơi cơ sở nhỏ"
> §3: "khoảng cách accuracy nhỏ, ví dụ dưới vài phần trăm"

Hiện có `worst_client_f1` (`api/main.py:362`, `export.py:32`, dashboard
`App.jsx:470`) — đó là **sàn**, không phải **độ lệch**. `ml/metrics.py` chỉ có
`confusion_matrix` và `classification_metrics`; không có hàm fairness nào.

Gap-vs-centralized hiện phải tự tính mắt thường khi đọc bảng comparison — trong khi
đề cương gọi đây là "kết quả cốt lõi của đề tài" (§8).

- [ ] C1. Thêm metric fairness tường minh: độ lệch chuẩn accuracy/Macro-F1 giữa các
      client, và khoảng cách best−worst.
- [ ] C2. Thêm `gap_vs_centralized` (accuracy và Macro-F1) vào comparison export —
      cột dẫn xuất, tính từ scenario centralized cùng seed.
- [ ] C3. Hiện `gap` và `fairness` trên dashboard cạnh worst-client F1.
- [ ] C4. Test cho cả hai nhóm metric.

---

## D. Backbone nhẹ và lượng tử hóa — P1/P2

> §4: "backbone nhẹ (**MobileNetV3 / EfficientNet-Lite**)"
> §5 GĐ4: "Backbone nhẹ, **lượng tử hóa** cho thiết bị biên"
> §8: "Mô hình tối ưu cho biên — mô hình client nhẹ **đã lượng tử hóa**"

`model.py:12-31` hỗ trợ `mobilenet_v2`, `mobilenet_v3_small`, `efficientnet_lite0`,
`resnet18`; `settings.py` mặc định `mobilenet_v3_small` — khớp đề cương.

Lưu ý trung thực: `_build_efficientnet_lite0` (`model.py:61-76`) thực chất dựng
`efficientnet_b0` của torchvision, **không phải EfficientNet-Lite thật** (Lite bỏ SE
block và swish). Cần ghi rõ trong báo cáo, hoặc đổi tên biến để không tuyên bố sai.

`ml/quantization.py` (215 dòng, tạo 06/08 10:20) — **không được gọi từ bất kỳ đâu,
không có test nào**. Xác minh bằng grep toàn repo: 0 tham chiếu ngoài chính nó.

- [ ] D1. Nối `quantization.py` vào một lệnh CLI hoặc script để tạo được checkpoint
      lượng tử hóa thật.
- [ ] D2. Test cho `quantise_model_static` / QAT path.
- [ ] D3. Đo và ghi lại: kích thước model trước/sau, Macro-F1 trước/sau, thời gian
      inference — §8 yêu cầu "mô hình đã lượng tử hóa chạy được trên thiết bị biên".
- [ ] D4. Ghi rõ `efficientnet_lite0` = `efficientnet_b0` trong docs, hoặc cài Lite thật.

---

## E. Differential privacy — P2 (đề cương ghi "tùy chọn")

> §5 GĐ4: "**tùy chọn** differential privacy"
> §7: "Quyền riêng tư — … **tùy chọn** định lượng thêm bằng differential privacy"

`fl/privacy.py` (7.6KB, tạo 06/08 10:20) — cũng **không được gọi từ đâu, không có
test**. Grep toàn repo: mọi hit "privacy" khác đều thuộc `data_profiles.py` /
`audit.py` (privacy boundary của dataset), không liên quan module DP này.

Vì đề cương ghi "tùy chọn", đây là P2 — nhưng đã tồn tại code chết thì phải hoặc nối
vào, hoặc test, hoặc xóa. Để nguyên là nợ kỹ thuật gây hiểu nhầm rằng đề tài đã có DP.

- [ ] E1. Test cho `compute_noise_multiplier` / clipping / budget accounting.
- [ ] E2. Nối vào client update path sau cờ bật/tắt, hoặc ghi rõ là chưa dùng.
- [ ] E3. Giữ D-014 — không được mô tả FL/DP là bảo đảm privacy hoàn chỉnh.

---

## F. Dữ liệu bổ sung — không bắt buộc

> §6: "**có thể** bổ sung PlantDoc (ảnh thực địa, nhiễu hơn) hoặc dữ liệu tự thu"
> §9: "bổ sung dữ liệu thực địa **nếu có điều kiện**"

Đề cương dùng "có thể" / "nếu có điều kiện" — không phải yêu cầu. PlantVillage 38 lớp
đã đáp ứng §6 phần bắt buộc: `profiles_index.json` xác nhận 54.305 ảnh, 38 lớp,
khớp "khoảng 54.000 ảnh, 38 lớp".

- [ ] F1. (Tùy chọn) PlantDoc — chỉ làm sau khi A–D xong.

---

## G. Đồng bộ tài liệu — P1

Docs mới cập nhật đến 31/07, chưa phản ánh việc chuyển sang 38 lớp hôm nay:

- [ ] G1. `ROADMAP`/`00_PROJECT_CONTEXT` vẫn nói phạm vi 10 lớp cà chua; chưa có
      entry nào cho 38 lớp, `alpha-100`, hay hai module `privacy`/`quantization`.
- [ ] G2. `10_TEST_REPORT` ghi "60 test"; thực tế **76 test đạt** (06/08, không
      tính `tests/api` vì `fastapi` chưa cài trong venv hiện tại).
- [ ] G3. `DECISION_LOG` cần entry mới cho: chuyển taxonomy 38 lớp (thay D-003),
      thêm `alpha-100` theo §6, và mở rộng thuật toán vượt D-007.
- [ ] G4. `data/flower-profiles-full.partial-20260806/` chỉ có `iid` rỗng — tàn dư
      lần chạy prep bị hỏng. Xác nhận rồi xóa.

---

## Thứ tự đề xuất

1. ~~**B1–B2**~~ — **XONG 06/08/2026.** `scaffold`/`moon` đã thực sự chạy đúng
   thuật toán và fail-fast nếu thiếu trạng thái. An toàn để chạy main study.
2. ~~**A1–A6**~~ — **XONG 06/08/2026.** Đủ 3 skew, 6 profile trên dữ liệu thật,
   audit `passed` cả 6, D-024 kiểm chứng lại trên đĩa.
3. **C1–C4** — metric mà §7/§8 gọi là kết quả cốt lõi.
4. **G1–G4** — đồng bộ docs trước khi khóa protocol.
5. **D, E, F** — GĐ4 trở đi, sau khi trục so sánh chính đã vững.

## Nguyên tắc giữ nguyên

- D-024: mọi profile chia sẻ **một** global test set duy nhất.
- D-028: artifact pilot mang `research_result_valid=false`, không lọt vào exporter.
- D-014: không mô tả FL/DP là bảo đảm privacy hoàn chỉnh.
- Không đánh `[x]` khi chưa có test/artifact chứng minh.
