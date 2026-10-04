# BÁO CÁO THỰC HÀNH LAB 16: CLOUD AI ENVIRONMENT SETUP (AWS)

## 1. Bảng Tổng Hợp Kết Quả Benchmark
| Metric | Kết quả |
|---|---|
| Thời gian load data | 2.4594 s |
| Thời gian training | 1.5871 s |
| Best iteration | 1 |
| AUC-ROC | 0.9517 |
| Accuracy | 0.9989 |
| F1-Score | 0.7273 |
| Precision | 0.6557 |
| Recall | 0.8163 |
| Inference latency (1 row) | 1.274 ms |
| Inference throughput (1000 rows) | 643,375.6 rows/s |

## 2. Nhận xét đánh giá (5 - 10 dòng)
1. **Thời gian huấn luyện (Training time):** Trên cấu hình CPU `t3.micro` khiêm tốn (2 vCPU, 1 GB RAM kết hợp Swap), mô hình LightGBM huấn luyện trên toàn bộ 227,845 mẫu chỉ mất **1.5871 giây**. Điều này chứng minh hiệu quả vượt trội của thuật toán histogram-based splitting trên CPU mà không nhất thiết phải đầu tư GPU đắt đỏ.
2. **Chất lượng mô hình (Model Quality):** Mô hình đạt chỉ số **AUC-ROC rất cao (0.9517)** và **Accuracy 99.89%**. Đối với bài toán dữ liệu mất cân bằng nghiêm trọng như gian lận thẻ tín dụng, chỉ số Recall đạt **81.63%** và F1-Score đạt **0.7273**, thể hiện khả năng phát hiện phần lớn các giao dịch gian lận thực tế với tỷ lệ báo động giả thấp.
3. **Tốc độ suy luận (Inference Speed):** Độ trễ cho 1 dòng dự đoán chỉ vỏn vẹn **1.274 ms** và thông lượng batch đạt tới **643,375 rows/giây**. Tốc độ này hoàn toàn đáp ứng tiêu chuẩn xử lý giao dịch thời gian thực (real-time scoring) tại cổng thanh toán.
4. **Hiệu quả chi phí (Cost Efficiency):** Việc triển khai mô hình dạng tabular/gradient boosting trên CPU node nhỏ trong mạng Private VPC vừa đảm bảo an toàn dữ liệu, vừa tối ưu hóa chi phí vận hành ở mức gần như bằng 0 (Free Tier).
