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
| `flower/client_app.py` | ✓ | ✓ | **✓ ĐÃ NỐI (H4)** | **✓ ĐÃ NỐI** | **✓ ĐÃ NỐI** |
| Whitelist API/CLI/config | ✓ | ✓ | ✓ | ✓ | ✓ |
| Test | ✓ | ✓ | **✓ 12 test** | **✓ 9 test** | **✓ 11 test** |

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

## H. Launcher main study không chạy được A và B — ĐÃ SỬA 07/08/2026

> §5 GĐ3: "Triển khai và so sánh FedProx, FedBN, SCAFFOLD, MOON"
> §6: "IID, Dirichlet nhiều mức, lệch số lượng, lệch đặc trưng"

A tạo 2 profile skew mới, B nối 3 thuật toán mới. Cả hai đều **không có đường vào
thực nghiệm**: `SCENARIOS` chỉ có 8 dòng (`centralized`, `local-only`, 4× fedavg,
2× fedprox) và `_resolve_profile_dir` chỉ nhận `iid`/`alpha-*`, raise `ValueError`
khi gặp `quantity-skew`/`feature-skew`. Bảng kết quả sẽ vẫn chỉ có FedAvg/FedProx
trên label skew, và không ai phát hiện cho tới lúc viết chương so sánh.

Khi sửa mới lộ ra **ba lỗi chặn nữa**, mỗi lỗi đều đủ để hỏng một lần chạy dài:

1. `parse_flower_log_evidence` raise `ValueError("algorithm must be 'fedavg' or
   'fedprox'")`. Một run `scaffold` sẽ train hết nhiều **giờ GPU** rồi fail ở đúng
   bước validate cuối cùng.
2. `partition_kind` được **đoán từ tên thư mục** (`"dirichlet" if name != "iid"`)
   trong cả `_run_local_only` lẫn `_run_federated`. Run quantity-skew sẽ tự ghi
   `partition_kind="dirichlet", dirichlet_alpha=0.0` — mâu thuẫn với chính
   `partition_summary.json` của nó và làm hỏng protocol lock.
3. **`fedbn` là no-op hoàn toàn trong đường Flower.**
   `fedbn_weighted_average_updates` chỉ gọi được từ `simulation.py`;
   `TrackedFedBN` là alias trần của `FedAvg`; client không làm gì riêng cho BN.
   Scenario `fedbn` sẽ sinh số của FedAvg dưới nhãn `fedbn` — đúng lỗi im lặng mà
   D-030 vừa vá cho `scaffold`/`moon`, lặp lại ở thuật toán thứ ba.

Ma trận sau khi sửa — **15 scenario**, xác minh bằng `--list-scenarios`:

| Trục | Scenario |
|---|---|
| Baseline | `CEN-MBV3`, `LOC-MBV3` |
| Label skew (fedavg) | `FL-IID-AVG`, `FL-A100-AVG`, `FL-A05-AVG`, `FL-A01-AVG` |
| FedProx | `FL-A05-PROX`, `FL-A01-PROX` |
| Thuật toán tại alpha-0.1 | `FL-A01-BN`, `FL-A01-SCAF`, `FL-A01-MOON` |
| Quantity skew | `FL-QTY-AVG`, `FL-QTY-PROX` |
| Feature skew | `FL-FEAT-AVG`, `FL-FEAT-BN` |

Ba thuật toán mới đều đặt tại `alpha-0.1` để **thuật toán là biến duy nhất**; mỗi
skew mới có một cặp fedavg-vs-thuật-toán-chịu-skew để đọc được chênh lệch.

- [x] H1. `SCENARIOS` 8 → 15, phủ đủ 5 thuật toán và 6 profile.
      `PROFILE_DIRECTORIES` thay ánh xạ hard-code trong `_resolve_profile_dir`.
- [x] H2. `_profile_alpha` → `_read_profile_metadata`, đọc `profile.json` thật.
      Quantity skew được chuyển tường minh từ `iid` sang `quantity_skew`
      (`test_quantity_skew_is_not_reported_as_iid`).
- [x] H3. `SUPPORTED_ALGORITHMS` + `_STRATEGY_LOG_NAMES` trong `smoke.py`; nhận cả
      tên upstream lẫn `Tracked*` vì `strategy.start` log `__class__.__name__`.
      Test lấy danh sách từ `get_args(Algorithm)` nên thuật toán thứ 6 sẽ tự fail.
- [x] H4. FedBN thành thật: client khôi phục BN riêng sau khi nạp global (cả
      nhánh train lẫn **evaluate**), báo `fedbn_local_bn_tensors`; `TrackedFedBN`
      raise từ round 2 nếu bằng 0. `batch_norm_parameter_names` dùng chung với
      simulator để `fedbn` không mang hai nghĩa khác nhau trong cùng luận văn.
- [x] H5. `algorithm_artifact_evidence` — chứng cứ ở **tầng artifact**, không chỉ
      log: SCAFFOLD phải đủ client mọi round; MOON/FedBN miễn round 1 nhưng bắt
      buộc từ round 2; giá trị 0 tính là "không tác dụng"; history thiếu round bị
      từ chối (run crash không được lọt qua bằng cách có ít round hơn).
- [x] H6. Hyperparameter `scaffold_server_lr`/`moon_temperature`/`moon_mu` ghi
      vào manifest + checkpoint + protocol lock, và được `validate_run_artifacts`
      đối chiếu với giá trị đã yêu cầu. `test_launcher_matches_the_server_recording_rule`
      dựng `flwr.app.Context` thật để hai bản cài đặt không trôi khỏi nhau.
- [x] H7. `--only` lặp lại được, thêm `--list-scenarios`. Theo D-028: chạy tập con
      thì `matrix_complete=false` và `research_result_valid=false` — summary của
      một phần ma trận không được tự nhận là main study.
- [x] H8. `fedbn`/`moon` với `rounds < 2` bị từ chối **trước** khi chạy, vì cả hai
      cần trạng thái round trước; nếu không sẽ fail sau khi đã tiêu hết GPU time.

**Bằng chứng:** `tests/system/test_main_study_matrix.py` (14 test, 55 subtest),
`tests/flower/test_flower_algorithms.py` +9 test FedBN, `tests/flower/test_flower_smoke.py`
+16 test. Toàn bộ suite **130 test đạt, 113 subtest** (`pytest tests --ignore=tests/api`;
`tests/api` không collect được vì venv hiện tại thiếu `fastapi`, có từ trước).
`_restore_local_batch_norm` được kiểm chứng bằng mutation test: chèn `return 0`
làm fail đúng 2 test, khôi phục xong 23 test đạt lại.

---

## C. Metric fairness và gap-vs-centralized — ĐÃ SỬA 07/08/2026

> §7: "báo cáo **KHOẢNG CÁCH** so với mô hình tập trung"
> §7: "**Tính công bằng** — độ lệch accuracy giữa cơ sở mạnh và yếu; hệ phân tán tốt
> không được bỏ rơi cơ sở nhỏ"
> §3: "khoảng cách accuracy nhỏ, ví dụ dưới vài phần trăm"

Trước đó chỉ có `worst_client_f1` — đó là **sàn**, không phải **độ lệch**: bốn client
ở 0,70/0,70/0,70/0,40 và bốn client ở 0,95/0,80/0,55/0,40 cho cùng một con số sàn,
trong khi cái thứ hai mới là liên đoàn đang bỏ rơi cơ sở nhỏ. Gap-vs-centralized thì
phải tự trừ bằng mắt khi đọc bảng, dù §8 gọi đây là "kết quả cốt lõi của đề tài".

- [x] C1. `client_fairness` trong `ml/metrics.py`: `std`, `spread` (best−worst),
      `coefficient_of_variation`, và khi có `num_examples` thì thêm `weighted_mean`
      + `size_advantage`. Trung bình là **không trọng số** và độ lệch chuẩn là của
      **quần thể** — xem "Hai quy ước" bên dưới.
- [x] C2. `gap_vs_centralized` là cột dẫn xuất trong comparison export
      (`gap_vs_centralized_accuracy`, `gap_vs_centralized_macro_f1`,
      `gap_baseline_run_id`), cùng bốn cột fairness. Ghép baseline theo
      **(seed, model)**, không phải seed đơn thuần.
- [x] C3. Dashboard có hai dòng mới cạnh Worst-Client F1: "Độ lệch giữa client"
      (std · spread) và "Khoảng cách vs tập trung", kèm chú thích dấu dương nghĩa là
      FL còn kém hơn. Mốc so sánh lấy từ `CROPFED_CENTRALIZED_BASELINE_RESULT`
      (đường dẫn phía server, HTTP không cấp được — D-019).
- [x] C4. 27 test mới: 11 ở `tests/unit/test_metrics.py`, 8 ở `tests/api/test_export.py`,
      8 ở `tests/api/test_new_endpoints.py`.

### Hai quy ước, cả hai đều ngược với metric tổng hợp

**Trung bình không trọng số.** Trọng số theo số mẫu chính là thứ che mất một cơ sở
nhỏ bị bỏ rơi, vì cơ sở bị bỏ rơi đúng là cơ sở ít dữ liệu. Test
`test_unweighted_mean_does_not_hide_an_abandoned_small_client` khóa lại tình huống
này: một client 10 mẫu ở 0,10 với ba client 1.000 mẫu ở 0,90 — `weighted_mean` vẫn
0,88 trông rất khỏe, còn `mean` tụt xuống 0,70 và `size_advantage` gọi tên hiệu ứng.

**Độ lệch chuẩn quần thể, không phải mẫu.** Bốn client này *là* toàn bộ liên đoàn,
không phải mẫu rút ra từ liên đoàn nào lớn hơn. Chia cho `n−1` sẽ thổi con số lên
~15% ở bốn client, làm run trông kém công bằng hơn thực tế.

### Dấu của gap

`gap = centralized − federated`, nên **dương luôn nghĩa là FL còn kém hơn**. Đảo dấu
sẽ đọc thành liên đoàn thắng mô hình tập trung — đúng luận điểm trung tâm của đề tài
nhưng quay ngược. Mutation test: đảo dấu làm fail 4 test ở cả ba lớp (metric, export,
API); đổi `pstdev` thành `stdev` làm fail 5 test.

### Ba cách cột gap có thể so nhầm — đã chặn

- **Khác backbone**: baseline MobileNetV2 với run MobileNetV3 sẽ báo chênh lệch kiến
  trúc thành cái giá của liên đoàn. Export ghép theo `(seed, model)`; API từ chối
  baseline không trùng `flower_model_name`.
- **Khác seed**: gap sẽ lẫn một phần chênh lệch seed. Run không có seed thì không bao
  giờ được ghép cặp.
- **Baseline pilot**: `research_result_valid=false` bị từ chối, đúng D-028 — không thể
  lấy con số mà exporter từ chối xuất bản làm kết quả đầu bảng.

Thiếu mốc thì cột để **trống**, không phải `0.0`: `0.0` sẽ khẳng định liên đoàn hòa
đúng bằng một baseline chưa từng chạy. Một client thì `std`/`spread` cũng để trống —
một client không phải một liên đoàn, và `0.0` sẽ đọc thành công bằng tuyệt đối.

### Bằng chứng

- `pytest tests` → **230 test đạt / 128 subtest**, 33,58s. Đây là lần đầu `tests/api`
  chạy được: mục C cài `fastapi` bằng `pip install -e ".[api,dev]"`, nên ghi chú
  "`tests/api` không collect được" ở các mục trước đã hết hiệu lực.
- `frontend/` chưa có hạ tầng test (không có vitest/jest trong `package.json`), nên
  C3 được phủ ở tầng API — test kiểm đúng dữ liệu mà dashboard render.
- Đính chính: dashboard nằm ở `frontend/src/App.jsx`, không phải
  `web/dashboard/src/App.jsx` như bản checklist trước ghi.

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

## G. Đồng bộ tài liệu — ĐÃ SỬA 07/08/2026

Docs dừng ở 31/07 trong khi phạm vi đã đổi hai lần. Nguy hiểm không phải ở chỗ
thiếu chữ mà ở chỗ **sai**: một phiên làm việc mới đọc `00_PROJECT_CONTEXT` sẽ
tưởng đề tài là 10 lớp cà chua, 3 profile và 4 thuật toán — rồi sửa code cho khớp
với tài liệu.

- [x] G1. `00_PROJECT_CONTEXT` §2/§6/§7/§8 đổi sang 38 lớp / 6 profile / 5 thuật
      toán; thêm §12.1 liệt kê mọi thứ vào sau mốc 0.1.0. `ROADMAP` có khối trạng
      thái đầu file, nói rõ câu "không mở rộng thuật toán" ở §1 đã bị D-030 thay.
      Hai module `privacy`/`quantization` được ghi ở **§9 (nâng cao)**, không phải
      §12 (đã có) — có file không đồng nghĩa với có năng lực.
- [x] G2. `10_TEST_REPORT` giữ con số 60 làm mốc lịch sử và thêm khối cập nhật:
      **230 test đạt / 128 subtest** (73 unit, 60 flower, 55 api, 42 system), cùng
      mục "Phạm vi bổ sung sau mốc 0.1.0" dẫn về A/B/C/H/K.
- [x] G3. `DECISION_LOG` thêm D-037 (taxonomy 38 lớp, đánh D-003 thành
      **Superseded** thay vì xóa) và D-038 (`alpha-100` + hai skew mới). Mở rộng
      thuật toán đã có ở D-030 từ 06/08, không cần entry trùng.
- [x] G4. Đã xóa `data/flower-profiles-full.partial-20260806/` sau khi đối chiếu —
      xem dưới.

### G4: thư mục partial **không** rỗng như bản kiểm kê ghi

Nó có 13 file, 28 MB, và là bản `iid` hoàn chỉnh. Nếu tin bản kiểm kê mà xóa thẳng
thì vẫn đúng kết quả nhưng sai lý do — nên đã đối chiếu trước, ghi ở
`artifacts/partial_profile_removal_check.json`:

- 97.752 dòng manifest (train/test/validation/pooled) khớp **từng dòng** với
  profile hiện hành sau khi cắt bỏ tiền tố máy: 0 dòng lệch;
- cả 8 file client (4 client × train/val) trùng đúng danh sách `image_id`;
- `partition_summary.json` giống hệt.

Khác biệt duy nhất: 97.752 đường dẫn đều là **tuyệt đối** (`F:\project\...`, tức
tiền D-034), và thư mục thiếu `profile.json` + `data_audit.json`. Đây là bản chụp
giữa chừng trước khi migrate, đã bị bản hiện hành thay thế hoàn toàn. `.gitignore:24`
vốn đã loại nó khỏi git, và không file nguồn/tài liệu nào tham chiếu tới nó.

---

## K. Artifact không chạy được trên máy khác — ĐÃ SỬA 07/08/2026

Không nằm trong bản kiểm kê gốc vì nó không phải khoảng trống so với đề cương mà
là khoảng trống so với **kế hoạch chạy**: train diễn ra trên máy GPU khác, còn
manifest thì ghi `F:\project\cropfed-thesis\data\...`. Sáu profile tốn nhiều giờ
phân hoạch sẽ vô dụng ngay khi rời laptop, và protocol lock khóa SHA-256 của
chính những manifest đó nên không thể "sửa path lúc chạy".

- [x] K1. Manifest lưu path **tương đối theo dataset root** cấp lúc chạy;
      `src/cropfed/data/paths.py` là định nghĩa duy nhất. Path tương đối mà thiếu
      root thì **raise**, không resolve theo CWD — 11 test ở
      `tests/unit/test_dataset_paths.py`, gồm test chứng minh cùng một manifest
      resolve được dưới cả `F:/project/.../data/raw` lẫn `/app/data/raw`.
- [x] K2. `--dataset-root` xuyên suốt prep → audit → dataloader → Flower;
      `dataset_root` thành tham số **bắt buộc** của `extend_data_profiles` (trước
      đây optional, và thiếu nó thì audit báo *mọi* ảnh `invalid_image`).
- [x] K3. `dataset-root` và `num-workers` **khai báo** trong
      `[tool.flwr.app.config]`; 6 test ở `tests/flower/test_run_config_paths.py`
      chốt lại, gồm test client và server phải đồng ý về cùng một root.
- [x] K4. `num_workers` đi từ `run_config` xuống `build_dataloader`. Mặc định 0
      (an toàn trên Windows) và phải nâng lên trên máy GPU — ghi trong §8.4
      `06_ROADMAP_AND_HANDOFF`.
- [x] K5. Migrate 6 profile có sẵn tại chỗ: `scripts/migrate_manifest_paths.py`,
      7 test ở `tests/system/test_manifest_migration.py`. Kết quả thật:
      **847.194 dòng, 0 lệch `image_id`**, `--verify` `status: passed`,
      `missing_files: 0`; audit lại profile `iid` mở được đủ **54.305 ảnh**,
      `errors=0`. `profiles_index.json` mang `manifest_paths:
      "relative_to_dataset_root"`, cả hai bất biến D-024 vẫn `true`.
- [x] K6. `scripts/rewrite_profile_paths.py` đánh dấu Superseded và raise ngay khi
      chạy — nó vẫn `replace("\\", "/")` nên sẽ làm hỏng bộ đã migrate. Giữ file
      vì `docs/10` và `docs/11` dẫn chiếu các run đã dùng nó.
- [x] K7. Trình sinh protocol lock cho cả 15 scenario:
      `scripts/generate_protocol_locks.py`, import lại `run_main_study.py` để
      dùng chung ma trận `SCENARIOS`. 10 test + 15 subtest ở
      `tests/system/test_protocol_lock_generator.py`, trong đó có test cho từng
      lock đi qua chính `validate_protocol_lock`.
- [x] K8. `DECISION_LOG` D-033/D-034/D-035; hướng dẫn bàn giao §8
      `06_ROADMAP_AND_HANDOFF.md`.

### Hai cái bẫy mà test đã khóa lại

`validate_protocol_lock` so khớp `config` **chính xác**, nên hai chi tiết dưới đây
sẽ khiến run fail *sau khi* đã đốt hết thời gian GPU:

- profile IID phải khóa `dirichlet_alpha = 0.0`, không phải `null` — runner ghi
  `alpha or 0.0` rồi server cast `float`;
- cả bốn hyperparameter thuật toán phải có mặt, giá trị 0 khi không áp dụng,
  đúng như `_algorithm_hyperparameters` sinh ra.

### Bằng chứng

- 164 test đạt / 128 subtest, 23,55s — đo lúc `tests/api` còn chưa collect được.
  Mục C sau đó cài `fastapi` (`pip install -e ".[api,dev]"`), nên con số hiện
  hành cho **toàn bộ** suite là 230 test đạt / 128 subtest.
- `artifacts/manifest_migration.json`, `artifacts/post_migration_audit_iid.json`.
- Báo cáo migration để ở `artifacts/`, **không** để trong bộ profile: nó nêu tên
  path local, còn bộ profile là thứ được bàn giao (D-025).

---

## Thứ tự đề xuất

1. ~~**B1–B2**~~ — **XONG 06/08/2026.** `scaffold`/`moon` đã thực sự chạy đúng
   thuật toán và fail-fast nếu thiếu trạng thái. An toàn để chạy main study.
2. ~~**A1–A6**~~ — **XONG 06/08/2026.** Đủ 3 skew, 6 profile trên dữ liệu thật,
   audit `passed` cả 6, D-024 kiểm chứng lại trên đĩa.
3. ~~**H1–H8**~~ — **XONG 07/08/2026.** Launcher chạy được đủ 5 thuật toán × 6
   profile; `fedbn` không còn là no-op. Trước H, A và B là năng lực không dùng được.
4. ~~**K1–K8**~~ — **XONG 07/08/2026.** Artifact và code chạy được trên máy GPU;
   protocol lock sinh được cho cả 15 scenario. Trước K, mọi thứ ở trên chỉ chạy
   trên đúng một cái laptop.
5. ~~**C1–C4**~~ — **XONG 07/08/2026.** Fairness là độ lệch chứ không còn là sàn;
   gap-vs-centralized thành cột thay vì phép trừ bằng mắt.
6. ~~**G1–G4**~~ — **XONG 07/08/2026.** Docs khớp lại với code trước khi khóa
   protocol. Tài liệu lệch không chỉ thiếu thông tin, nó chỉ sai đường.
7. **D, E, F** — GĐ4 trở đi, sau khi trục so sánh chính đã vững.

## Nguyên tắc giữ nguyên

- D-024: mọi profile chia sẻ **một** global test set duy nhất.
- D-028: artifact pilot mang `research_result_valid=false`, không lọt vào exporter.
- D-014: không mô tả FL/DP là bảo đảm privacy hoàn chỉnh.
- D-025: bộ profile bàn giao không chứa path local; báo cáo có path để ở `artifacts/`.
- D-033: thiếu dataset root thì **raise**, không bao giờ resolve theo CWD.
- Protocol lock sinh **trước** run mà nó ràng buộc, không sinh từ output của run.
- D-036: fairness dùng trung bình **không trọng số** + độ lệch chuẩn **quần thể**;
  `gap = centralized − federated` nên dương nghĩa là FL còn kém hơn.
- Thiếu mốc so sánh thì để **trống**, không bao giờ điền `0.0`.
- Không đánh `[x]` khi chưa có test/artifact chứng minh.
