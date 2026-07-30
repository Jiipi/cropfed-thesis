# 07 — Tài liệu tham khảo đã xác minh

Ngày kiểm tra liên kết gần nhất: **30/07/2026**. Ưu tiên bài gần đây và tài liệu chính thức; các công trình cũ hơn được giữ khi là nguồn gốc thuật toán/dataset.

## A. Nghiên cứu gần bài toán

1. D. Mamba Kabala, A. Hafiane, L. Bobelin, et al., “Image-based crop disease detection with federated learning,” *Scientific Reports*, vol. 13, article 19220, 2023. [DOI và bài toàn văn](https://www.nature.com/articles/s41598-023-46218-5).  
   Nguồn gần nhất với đề tài: dùng PlantVillage, crop disease classification, CNN/ViT và FL.

2. T. Wei, Z. Chen, Z. Huang, and X. Yu, “Benchmarking In-the-Wild Multimodal Plant Disease Recognition and A Versatile Baseline,” *ACM Multimedia*, 2024. [Bài báo](https://arxiv.org/html/2408.03120v1), [trang dự án](https://tqwei05.github.io/PlantWild/).  
   Nguồn cho hạn chế domain gap và hướng đánh giá in-the-wild nâng cao.

3. X. Wu et al., “IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition,” *CVPR*, 2019. [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_IP102_A_Large-Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.html).  
   Nguồn phụ cho nhận dạng sâu hại; không phải taxonomy chính của MVP.

## B. Dataset và computer vision

4. S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using Deep Learning for Image-Based Plant Disease Detection,” *Frontiers in Plant Science*, vol. 7, 2016. [Bài toàn văn](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full).  
   Nguồn học thuật cho PlantVillage: 54.306 ảnh, 38 lớp, 14 cây.

5. PlantVillage dataset repository. [GitHub](https://github.com/spmohanty/plantvillage-dataset).  
   Nguồn cấu trúc dữ liệu và tải dataset; cần kiểm tra điều khoản trước phân phối.

6. M. Sandler et al., “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” *CVPR*, 2018. [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html).  
   Nguồn kiến trúc model chính.

7. Torchvision, `mobilenet_v2` and `MobileNet_V2_Weights`, tài liệu stable. [Tài liệu chính thức](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html).  
   Nguồn API, ImageNet weights và normalization.

## C. Federated Learning và Non-IID

8. H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” *AISTATS/PMLR 54*, 2017. [PMLR](https://proceedings.mlr.press/v54/mcmahan17a.html).  
   Công trình nền tảng của Federated Averaging.

9. T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated Optimization in Heterogeneous Networks,” *MLSys*, 2020. [MLSys](https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html).  
   Công trình FedProx cho statistical/system heterogeneity.

10. S. P. Karimireddy et al., “SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,” *ICML/PMLR 119*, 2020. [PMLR](https://proceedings.mlr.press/v119/karimireddy20a.html).  
    Thuật toán nâng cao xử lý client drift; không bắt buộc MVP.

11. S. J. Reddi et al., “Adaptive Federated Optimization,” *ICLR*, 2021. [OpenReview](https://openreview.net/forum?id=LkFG3lB13U5).  
    Nguồn FedAdagrad/FedAdam/FedYogi cho phần mở rộng.

12. P. Kairouz et al., “Advances and Open Problems in Federated Learning,” *Foundations and Trends in Machine Learning*, 2021. [arXiv](https://arxiv.org/abs/1912.04977).  
    Tổng quan nền tảng về optimization, privacy, fairness và system challenges.

## D. Tài liệu triển khai chính thức

13. Flower Framework, “Flower Architecture,” v1.32.x. [Tài liệu chính thức](https://flower.ai/docs/framework/explanation-flower-architecture.html).

14. Flower Framework, “Quickstart PyTorch.” [Tài liệu chính thức](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html).  
    Nguồn Message API, `ArrayRecord`, `ClientApp`, `ServerApp` và `flwr run`.

15. Flower Datasets, “Use Partitioners.” [Tài liệu chính thức](https://flower.ai/docs/datasets/tutorial-use-partitioners.html).  
    Nguồn cho IID/Dirichlet và ý nghĩa alpha.

16. Flower Framework, `FedAvg` strategy API. [Tài liệu chính thức](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html).

17. Flower Framework, `FedProx` strategy API. [Tài liệu chính thức](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedProx.html).  
    Tài liệu xác nhận proximal term phải được tính ở client.

18. FastAPI, “SQL (Relational) Databases.” [Tài liệu chính thức](https://fastapi.tiangolo.com/tutorial/sql-databases/).

19. FastAPI, “CORS.” [Tài liệu chính thức](https://fastapi.tiangolo.com/tutorial/cors/).

20. Docker, “Control startup and shutdown order in Compose.” [Tài liệu chính thức](https://docs.docker.com/compose/how-tos/startup-order/).

## E. Quyền riêng tư

21. NIST, “Privacy Attacks in Federated Learning,” 24/01/2024. [NIST](https://www.nist.gov/blogs/cybersecurity-insights/privacy-attacks-federated-learning).  
    Nguồn để tránh tuyên bố sai rằng update không thể rò rỉ dữ liệu.

## Quy tắc trích dẫn trong báo cáo

- Dùng bài báo gốc cho định nghĩa thuật toán, không dùng blog thay thế.
- Dùng tài liệu Flower/PyTorch cho API/version, không dùng tutorial không chính thức.
- Gắn nguồn ngay sau số liệu dataset.
- Không chép accuracy từ nghiên cứu khác làm target của đề tài vì split/model khác.
- Mỗi claim thực nghiệm của đồ án phải trỏ tới artifact do chính hệ thống tạo.

