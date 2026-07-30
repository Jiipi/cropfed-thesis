# Quy ước thư mục dữ liệu

Ảnh gốc không được commit hoặc đưa vào gói bàn giao.

Snapshot local đã kiểm chứng ngày 30/07/2026 lấy từ
`https://github.com/spMohanty/PlantVillage-Dataset` tại commit
`7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`. Subset tomato có 18.160 ảnh/10
lớp. Ba profile hiện nằm ở `data/flower-profiles/`; `profiles_index.json` khóa
manifest/audit SHA-256. Xem kết quả tại
[`../docs/11_PLANTVILLAGE_PILOT_REPORT.md`](../docs/11_PLANTVILLAGE_PILOT_REPORT.md).

Sau khi chạy `cropfed prepare-data`, cấu trúc mong đợi:

```text
data/
├── raw/                         # tùy chọn; bị gitignore
├── processed/
│   ├── train_manifest.csv       # toàn bộ pool huấn luyện, dùng cho centralized
│   ├── test_manifest.csv        # global test giữ riêng
│   └── data_audit.json          # hash/overlap/corruption report, không chứa path ảnh
└── clients/
    ├── partition_summary.json
    ├── client_0/
    │   ├── train_manifest.csv
    │   └── val_manifest.csv
    ├── client_1/
    ├── client_2/
    └── client_3/
```

Manifest lưu đường dẫn tuyệt đối tới ảnh. Nếu chuyển project sang máy khác, cần tạo lại manifest bằng cùng seed thay vì sửa đường dẫn thủ công.

Sau khi tạo từng partition, chạy `cropfed audit-data` trước khi huấn luyện.
Audit phải đạt và report phải được lưu cùng checksum/config của thí nghiệm.
