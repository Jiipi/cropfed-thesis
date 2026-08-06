1. Hiểu đúng tên đề tài
Thành phần trong tên	Ý nghĩa triển khai
Nghiên cứu	Tổng quan CNN, FL, Non-IID, FedAvg, FedProx, quyền riêng tư và các nghiên cứu liên quan
Xây dựng hệ thống	Có server, client, web dashboard, cơ sở dữ liệu, quy trình huấn luyện và dự đoán
Học liên kết	Ảnh gốc nằm tại client; server tổng hợp cập nhật mô hình
Phát hiện sâu bệnh qua ảnh	Nhận ảnh và xác định ảnh thuộc lớp khỏe, bệnh hoặc sâu hại nào
Dữ liệu phân tán	Dữ liệu được chia cho nhiều cơ sở/client
Không đồng nhất	Tỷ lệ lớp, số lượng và điều kiện ảnh khác nhau giữa các client
Cơ sở nông nghiệp	Được mô phỏng bằng các client Cross-Silo; chưa cần trang trại thật
“Phát hiện” không bắt buộc phải là Object Detection

Trong Computer Vision, “object detection” thường có nghĩa là định vị đối tượng bằng bounding box. Nhưng trong nghiên cứu bệnh cây, từ “detection” thường được dùng theo nghĩa rộng là nhận biết bệnh.

Ví dụ bài báo có tên Image-based crop disease detection with federated learning, nhưng phần phương pháp xác định rõ họ thực hiện crop disease classification bằng CNN và ViT. Scientific Reports, 2023.

Vì vậy, bạn có thể giữ nguyên tên đề tài và định nghĩa trong báo cáo:

Trong phạm vi đề tài, “phát hiện sâu bệnh cây trồng qua ảnh” được thực hiện ở mức ảnh — image-level recognition/classification. Hệ thống nhận một ảnh cây trồng và dự đoán ảnh thuộc trạng thái khỏe mạnh, một lớp bệnh hoặc một lớp sâu hại đã được huấn luyện. Đề tài không thực hiện định vị bounding box hoặc phân đoạn vùng tổn thương.

Nên đưa nguyên tắc này vào phần phạm vi và nhờ giảng viên hướng dẫn xác nhận để tránh hội đồng hiểu “phát hiện” là bắt buộc dùng YOLO.

2. Bài toán thực tế được điều chỉnh đúng theo tên đề tài

Mỗi cơ sở nông nghiệp có thể thu thập ảnh cây trồng khác nhau:

Cơ sở A trồng nhiều cà chua và chủ yếu gặp bệnh nấm.
Cơ sở B gặp nhiều bệnh do virus.
Cơ sở C có nhiều ảnh cây khỏe.
Cơ sở D gặp nhiều sâu, nhện hoặc điều kiện chụp khác.
Số ảnh và thiết bị chụp của các cơ sở cũng khác nhau.

Nếu từng cơ sở tự huấn luyện thì dữ liệu quá ít và mô hình dễ thiên lệch. Nếu đưa toàn bộ ảnh về server thì phát sinh:

Chi phí truyền và lưu trữ.
Rủi ro lộ thông tin mùa vụ, vị trí hoặc quy trình sản xuất.
Vấn đề quyền sở hữu và kiểm soát dữ liệu.
Các cơ sở có thể không đồng ý chia sẻ ảnh thô.

Hệ thống FL giải quyết bằng cách:

Mỗi client huấn luyện tại chỗ.
Chỉ gửi model update và metrics lên server.
Server tổng hợp thành global model.
Global model được gửi lại cho các cơ sở.
Mỗi cơ sở có thể phát hiện các lớp sâu bệnh trên ảnh mới.

Lưu ý: FL giảm nhu cầu tập trung ảnh nhưng không bảo mật tuyệt đối; cập nhật mô hình vẫn có thể bị phân tích. NIST, 2024.

3. Mục tiêu đề tài đã chỉnh sửa
Mục tiêu tổng quát

Nghiên cứu và xây dựng hệ thống Federated Learning cho phép nhiều cơ sở nông nghiệp phối hợp huấn luyện mô hình phát hiện sâu bệnh cây trồng qua ảnh trong điều kiện dữ liệu phân tán và Non-IID, đồng thời không yêu cầu truyền ảnh huấn luyện gốc lên server trung tâm.

Mục tiêu cụ thể
Nghiên cứu CNN, transfer learning và phân loại ảnh sâu bệnh cây trồng.
Nghiên cứu Cross-Silo FL, FedAvg, FedProx và Non-IID.
Xây dựng tập dữ liệu phân tán giả lập cho nhiều cơ sở.
Xây dựng centralized model làm baseline.
Xây dựng local-only model cho từng cơ sở làm baseline.
Xây dựng quy trình FL với FedAvg.
Dùng FedProx để xử lý dữ liệu không đồng nhất.
Đánh giá ảnh hưởng của mức độ Non-IID.
So sánh centralized, local-only và federated.
Xây dựng web dashboard quản lý thí nghiệm.
Cung cấp chức năng dự đoán sâu bệnh trên ảnh mới.
Đánh giá hiệu năng, thời gian và chi phí giao tiếp.
4. Câu hỏi nghiên cứu phù hợp với đề tài chính thức
Mã	Câu hỏi
RQ1	Federated Learning có xây dựng được mô hình phát hiện sâu bệnh từ dữ liệu phân tán mà không truyền ảnh gốc không?
RQ2	Dữ liệu Non-IID ảnh hưởng thế nào đến Accuracy, Macro-F1 và tốc độ hội tụ?
RQ3	FedProx có ổn định hoặc hiệu quả hơn FedAvg khi mức độ Non-IID tăng không?
RQ4	Global model có tốt hơn các local-only model của từng cơ sở không?
RQ5	Global model chênh lệch thế nào so với centralized model?
RQ6	Mô hình nhẹ có giảm thời gian và dung lượng truyền mà vẫn giữ hiệu quả phát hiện không?
Giả thuyết nghiên cứu
H1: Non-IID càng mạnh thì FedAvg càng dễ dao động hoặc giảm hiệu quả.
H2: FedProx ổn định hơn FedAvg trong kịch bản Non-IID mạnh.
H3: Global FL model tốt hơn phần lớn local-only model.
H4: Centralized có thể tốt hơn FL, nhưng FL không cần tập trung ảnh gốc.
H5: MobileNetV2 tạo cân bằng tốt giữa Macro-F1, thời gian và dung lượng model.

Không được viết các giả thuyết này thành kết luận trước khi chạy thực nghiệm.

5. Phạm vi “sâu bệnh cây trồng” hợp lý
Phương án khuyến nghị cho MVP

Sử dụng phần Tomato của PlantVillage, gồm:

Tomato healthy.
Bacterial spot.
Early blight.
Late blight.
Leaf mold.
Septoria leaf spot.
Target spot.
Tomato mosaic virus.
Tomato yellow leaf curl virus.
Two-spotted spider mite.

Như vậy hệ thống có:

Lớp cây khỏe.
Bệnh do vi khuẩn.
Bệnh do nấm.
Bệnh do virus.
Một lớp sinh vật gây hại là nhện đỏ.

Phạm vi này xoay đúng quanh “sâu bệnh cây trồng” hơn phương án chỉ làm ba lớp bệnh.

PlantVillage có tổng cộng 54.306 ảnh, 38 lớp của 14 loại cây trồng và 26 bệnh/trạng thái khỏe. Nguồn dữ liệu gốc.

Nếu giảng viên bắt buộc có nhiều loại cây

Có thể mở rộng sang bốn nhóm giống nghiên cứu năm 2023:

Tomato.
Apple.
Corn.
Grape.

Nhưng phương án này làm tăng:

Số lớp.
Kích thước mô hình.
Số thí nghiệm.
Thời gian huấn luyện.
Độ khó xây dựng label mapping.

Vì vậy chỉ nên làm sau khi phiên bản một cây chạy ổn.

Nếu giảng viên bắt buộc phải phát hiện côn trùng thật

Khi đó mới xem xét IP102. Dataset có hơn 75.000 ảnh thuộc 102 loại sâu hại và khoảng 19.000 ảnh có bounding box. Tuy nhiên đây là dataset long-tail rất khó, không nên ghép toàn bộ với PlantVillage trong MVP. IP102 – CVPR 2019.

Phương án an toàn hơn là chọn một nhóm nhỏ sâu hại liên quan đến một cây cụ thể và dùng một mô hình riêng trong phiên bản nâng cao.

6. Dữ liệu chính và dữ liệu đánh giá thực tế
Mục đích	Dataset	Vai trò
Huấn luyện MVP	PlantVillage Tomato	Dữ liệu chính
Kiểm tra ảnh tự nhiên	PlantDoc	External test nếu ánh xạ được nhãn
Kiểm tra miền thực địa khó hơn	PlantWild	Full version
Chuyển sang cây lúa	Paddy Doctor	Phương án thay thế
Nhận diện sâu hại	IP102	Nâng cao

PlantVillage chủ yếu có ảnh trong môi trường được kiểm soát. PlantDoc và PlantWild có nền, góc chụp và ánh sáng tự nhiên hơn. PlantWild công bố 18.542 ảnh với 89 loại bệnh; nghiên cứu cũng chỉ ra khoảng cách lớn giữa dữ liệu phòng thí nghiệm và ảnh thực địa. PlantWild, ACM Multimedia 2024.

Do đó, báo cáo phải ghi rõ:

Mô hình được đánh giá chính trên dữ liệu công khai giả lập. Kết quả cao trên PlantVillage chưa chứng minh hệ thống đạt hiệu quả tương đương ngoài đồng ruộng.

7. Mô phỏng các cơ sở nông nghiệp

MVP dùng bốn client:

Client	Mô phỏng
Cơ sở A	Nhiều ảnh bệnh nấm
Cơ sở B	Nhiều ảnh bệnh virus
Cơ sở C	Nhiều ảnh cây khỏe và bệnh vi khuẩn
Cơ sở D	Nhiều ảnh spider mite và một số bệnh khác

Không nên chia thủ công cố định hoàn toàn vì có thể quá cực đoan. Nên dùng Dirichlet:

IID: phân bố gần giống nhau.
Non-IID vừa: α=0.5.
Non-IID mạnh: α=0.1.

Ngoài label skew, full version có thể thêm:

Quantity skew: số ảnh mỗi cơ sở khác nhau.
Feature skew: ánh sáng, độ mờ, màu và nền khác nhau.
Client dropout: một cơ sở không tham gia một số round.

Chỉ phần train được partition. Global test set phải tách trước và không thay đổi giữa các thuật toán.

8. Chức năng hệ thống đã điều chỉnh
Chức năng bắt buộc
Quản lý tài khoản và quyền cơ bản.
Đăng ký cơ sở/client.
Xem trạng thái kết nối client.
Xem số lượng và phân bố lớp tại client dưới dạng thống kê.
Không xem hoặc tải ảnh riêng tư từ server.
Tạo thí nghiệm:
Mô hình.
FedAvg/FedProx.
Số round.
Local epoch.
IID/Non-IID.
Khởi chạy huấn luyện liên kết.
Gửi global model đến client.
Huấn luyện local model.
Tổng hợp model update.
Đánh giá global và local model.
Theo dõi Accuracy, Macro-F1, loss, thời gian và round.
Lưu checkpoint.
So sánh Centralized, Local-only, FedAvg và FedProx.
Xuất kết quả thực nghiệm.
Dự đoán ảnh mới tại client.
Kết quả dự đoán nên gồm
Loại cây.
Nhãn dự đoán.
Nhóm: khỏe/bệnh/sâu hại.
Top-3 lớp có xác suất cao nhất.
Confidence.
Phiên bản model.
Cảnh báo “chỉ hỗ trợ tham khảo”.
Nâng cao
Grad-CAM.
Dữ liệu thực địa.
Secure Aggregation hoặc Differential Privacy.
Client dropout.
Phát hiện cập nhật bất thường.
Mô hình riêng cho sâu hại IP102.
Triển khai client trên nhiều máy vật lý.

Không nên tự động đề xuất thuốc hoặc liều lượng hóa chất.

9. Kiến trúc đúng với đề tài
Đang tải sơ đồ...

Server trung tâm không lưu ảnh huấn luyện của cơ sở. Nó chỉ lưu:

Model update/global model.
Số lượng mẫu tham gia.
Metrics.
Cấu hình experiment.
Log và checkpoint.

Ảnh người dùng gửi để dự đoán nên được xử lý tại client. Nếu làm web inference tập trung để demo, phải mô tả đây là luồng dự đoán tự nguyện, tách biệt với dữ liệu huấn luyện riêng tư.

10. Thiết kế thực nghiệm đúng trọng tâm
Mô hình
Chính: MobileNetV2.
Full version: thêm ResNet18 hoặc ResNet50.
Transfer learning từ ImageNet.
Ảnh resize 224×224.

Nghiên cứu FL về bệnh cây năm 2023 nhận thấy ResNet50 đạt kết quả tốt trong nhiều cấu hình, trong khi MobileNetV2 phù hợp hơn khi quan tâm tài nguyên và chi phí tính toán. Nguồn.

Phương pháp so sánh
Phương pháp	Ý nghĩa
Local-only	Mỗi cơ sở tự học, không cộng tác
Centralized	Gộp dữ liệu; mốc tham chiếu trên
FedAvg IID	Kiểm tra FL trong điều kiện thuận lợi
FedAvg Non-IID	Đo ảnh hưởng của dữ liệu không đồng nhất
FedProx Non-IID	Kiểm tra phương pháp giảm client drift
Ma trận MVP
Một dataset/cây trồng.
Ba phân bố: IID, Non-IID vừa, Non-IID mạnh.
Bốn phương pháp.
Ba random seed.

Tổng khoảng 36 run. Đây là khối lượng tương đối lớn nhưng vẫn kiểm soát được.

11. Tiêu chí đánh giá bổ sung cho “sâu bệnh”

Ngoài Accuracy, Precision, Recall và Macro-F1, nên bổ sung:

Recall từng lớp sâu bệnh.
F1 của lớp spider mite.
F1 của nhóm disease.
Tỷ lệ bỏ sót ảnh có hại thành healthy.
Confusion matrix.
F1 trung bình từng cơ sở.
Worst-client F1.
Số round để hội tụ.
Thời gian mỗi round.
Dung lượng upload/download.
Kích thước global model.
Thời gian dự đoán một ảnh.

Lỗi nguy hiểm nhất là ảnh bị bệnh/sâu hại nhưng hệ thống dự đoán thành healthy. Vì vậy Recall của các lớp gây hại cần được thảo luận, không chỉ nhìn Accuracy tổng.

12. Phiên bản tối thiểu bám sát đề tài chính thức
Giữ nguyên tên đề tài.
Phát hiện ở mức phân loại toàn ảnh.
Một cây trồng: tomato.
Mười lớp gồm khỏe, bệnh và spider mite.
PlantVillage.
4 cơ sở giả lập.
IID và hai mức Non-IID.
MobileNetV2.
Local-only, Centralized, FedAvg và FedProx.
Dashboard quản lý huấn luyện.
Dự đoán ảnh tại client.
Đánh giá Accuracy, Macro-F1, Recall lớp sâu bệnh, time, rounds và bytes.
Không bounding box, segmentation, IoT hoặc mobile app.

Độ khó: khoảng 7/10.

13. Phiên bản đầy đủ bám sát đề tài

Bao gồm bản tối thiểu và thêm:

Nhiều cây trồng.
PlantDoc hoặc PlantWild để external test.
Một mô hình thứ hai.
Quantity skew và feature skew.
Partial participation/client dropout.
Grad-CAM.
Một nhóm côn trùng chọn lọc từ IP102.
Secure Aggregation hoặc Differential Privacy.
Triển khai hai hoặc ba máy vật lý.

Độ khó: 8,5–9/10.

14. Những phần từ phân tích trước vẫn giữ nguyên

Các nội dung sau vẫn đúng và không cần thay đổi:

Kiến thức ML/DL/CNN cần học.
FedAvg, FedProx, SCAFFOLD và FedAdam.
Cách đánh giá Centralized–Local–Federated.
FastAPI, React, PostgreSQL, PyTorch và Flower.
Mô hình server–client.
Yêu cầu không truyền ảnh gốc.
Kế hoạch 16 tuần.
Các rủi ro về data leakage, domain shift và reproducibility.
Sản phẩm bàn giao và cấu trúc báo cáo.

Phần cần sửa chủ yếu là:

Không đề xuất đổi tên.
Không thu hẹp tên đề tài thành chỉ “phân loại bệnh”.
Đưa cả nhóm sâu hại vào class scope ở mức hợp lý.
Giải thích “phát hiện” là image-level recognition.
Điều chỉnh output thành khỏe/bệnh/sâu hại.
Thêm đánh giá Recall của các lớp gây hại.
Câu chốt nên ghi trong đề cương

Đề tài giữ nguyên định hướng phát hiện sâu bệnh cây trồng qua ảnh. Trong phạm vi triển khai, bài toán được mô hình hóa dưới dạng phân loại ảnh đa lớp, trong đó mỗi ảnh được xác định là cây khỏe mạnh hoặc thuộc một lớp sâu bệnh được hỗ trợ. Các ảnh huấn luyện được phân tán không đồng nhất tại nhiều client đại diện cho các cơ sở nông nghiệp; quá trình huấn luyện sử dụng Federated Learning để tạo mô hình toàn cục mà không tập trung ảnh gốc.