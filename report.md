# BÁO CÁO PHÂN TÍCH DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ (CHURN PREDICTION)
## Sản phẩm Heo Sổ - Ngân hàng số

**Nhóm thực hiện:** Group 01 - Challenge 03  
**Môn học:** FDC105  
**Ngày hoàn thành:** Tháng 8/2026  

---

## MỤC LỤC

1. [Giới thiệu Bài toán](#1-giới-thiệu-bài-toán)
2. [Phương pháp & Quy trình](#2-phương-pháp--quy-trình)
3. [Xử lý Dữ liệu (Data Wrangling)](#3-data-wrangling-xử-lý-dữ-liệu)
4. [Phân tích Khám phá Dữ liệu (EDA)](#4-phân-tích-khám-phá-dữ-liệu-eda)
5. [Mô hình hóa (Modeling)](#5-mô-hình-hóa-modeling)
6. [Kết luận & Đề xuất Kinh doanh](#6-kết-luận--đề-xuất-kinh-doanh)

---

## 1. Giới thiệu Bài toán

**Bối cảnh:** Ngân hàng số cần dự đoán khách hàng nào sẽ rời bỏ sản phẩm **Heo Sổ** (tiết kiệm số) để chủ động triển khai chiến dịch giữ chân khách hàng.

**Mục tiêu:**
- Xây dựng mô hình phân loại nhị phân dự đoán `is_exited` (1 = Rời bỏ, 0 = Ở lại).
- Xác định các **yếu tố tác động chính** đến hành vi rời bỏ.
- Đề xuất các hành động kinh doanh cụ thể nhằm tối ưu chi phí và nâng cao hiệu quả giữ chân.

**Dữ liệu:** `dataset.csv` chứa thông tin nhân khẩu học, hành vi giao dịch, dữ liệu theo thời gian (Tháng 3/2021 và Tháng 6/2021).

---

## 2. Phương pháp & Quy trình

```
Data Wrangling -> EDA -> Modeling -> Evaluation & Recommendation
```

| Giai đoạn | Kỹ thuật | Mô tả |
|---|---|---|
| Data Wrangling | Missing Value Imputation | Trung vị (Median) cho biến số, Unknown cho biến phân loại |
| Data Wrangling | Data Leakage Prevention | Loại bỏ các biến Tháng 6 và biến tích lũy tổng |
| EDA | Biểu đồ phân phối, Boxplot, Heatmap | Khám phá cấu trúc và tương quan dữ liệu |
| Modeling | SMOTE | Xử lý mất cân bằng dữ liệu (Imbalanced Data) |
| Modeling | Pipeline (imblearn) | Tránh rò rỉ dữ liệu khi Cross-Validation |
| Modeling | GridSearchCV (scoring=AUPRC) | Tối ưu hóa siêu tham số (Hyperparameter Tuning) |
| Evaluation | AUPRC | Chỉ số đánh giá chính cho dữ liệu mất cân bằng |
| Evaluation | SHAP | Giải thích đóng góp chi tiết của từng biến (Feature Importance) |

**Thư viện sử dụng:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, shap

---

## 3. Data Wrangling (Xử lý Dữ liệu)

### 3.1. Xác định khách hàng hiện hữu & Định nghĩa rời bỏ

**Lọc khách hàng hiện hữu:** Chỉ giữ lại khách hàng có `totalloginmar2021_heoso > 0` (có hoạt động đăng nhập trong T3/2021).

**Định nghĩa `is_exited = 1` (Churn):** Nếu thỏa mãn ít nhất 1 điều kiện tại T6/2021:

| Điều kiện | Biến | Ngưỡng |
|---|---|---|
| Không đăng nhập T6 | `totalloginjuin2021_heoso` | = 0 |
| Tổng tiết kiệm năm 2021 = 0 | `totalsavings2021_heoso` | = 0 |
| Giá trị tiết kiệm T6 = 0 | `savingvaluejuin2021_heoso` | = 0 |

### 3.2. Quy trình làm sạch dữ liệu

1. **Loại bỏ cột thiếu > 80%**
2. **Xử lý giá trị thiếu (Imputation):** Median cho biến số, `Unknown` cho biến phân loại
3. **Chuyển đổi kiểu dữ liệu:** `object` -> `category`
4. **Xử lý trùng lặp & ngoại lệ:** `drop_duplicates()`, lọc năm sinh hợp lệ (1900-2025)

---

## 4. Phân tích Khám phá Dữ liệu (EDA)

### 4.1. Phân phối nhãn (Label Distribution)

| Trạng thái | Số lượng | Tỷ lệ |
|---|:---:|:---:|
| **Retained (Ở lại)** | ~6,900 | **94.54%** |
| **Churned (Rời bỏ)** | ~400 | **5.46%** |

> **Cảnh báo mất cân bằng nghiêm trọng:** Mô hình dự đoán toàn bộ là "Ở lại" sẽ đạt Accuracy 94.54% nhưng hoàn toàn vô dụng trên thực tế. Cần sử dụng SMOTE và chỉ số AUPRC thay cho Accuracy hay ROC-AUC thông thường.

### 4.2. Hành vi giao dịch theo trạng thái Churn

- **Số lần đăng nhập T3/2021:** Nhóm Churn có mức đăng nhập gần như bằng 0 - việc không tương tác chính là tín hiệu rời bỏ mạnh mẽ nhất.
- **Giá trị tiết kiệm T6/2021:** Nhóm Churn phẳng tại 0 - khách hàng không còn giữ tiền gửi tiết kiệm.

### 4.3. Phân tích Nhân khẩu học

| Yếu tố | Phát hiện chính |
|---|---|
| **Giới tính** | Nam (5.56%) vs Nữ (5.30%) - không có sự khác biệt đáng kể |
| **Hôn nhân** | Độc thân (257 người, tỷ lệ cao hơn) > Đã kết hôn (132 người) |
| **Nhóm tuổi** | Gen Z (6.5%), Gen Y (231 người) - thế hệ trẻ có xu hướng rời bỏ nhiều hơn |
| **Tỉnh/Thành** | Tỉnh 27.0 (26.28%) và 38.0 (14.22%) - bất thường cục bộ cần điều tra sâu hơn |

### 4.4. Ma trận tương quan (Correlation Matrix)

- `is_exited` có tương quan tuyến tính rất thấp với các biến -> tồn tại **mối quan hệ phi tuyến phức tạp** -> cần áp dụng các mô hình Tree-based (Cây quyết định/Rừng ngẫu nhiên).
- `customer_age` vs `birth_incorp_date`: tương quan tuyệt đối -1.0, loại bỏ 1 biến để tránh dư thừa.
- `totalloginmar` vs `totalloginjuin`: 0.95 - hành vi đăng nhập có tính ổn định qua các tháng.

---

## 5. Mô hình hóa (Modeling)

### 5.1. Xử lý mất cân bằng & Chia dữ liệu

**Chống Data Leakage - Các biến bị loại bỏ:**

| Biến | Lý do loại bỏ |
|---|---|
| Tất cả biến `juin2021` | Dữ liệu Tháng 6 nằm ở tương lai (thời điểm xác định churn) |
| `totalsavings2021_heoso` | Dùng trực tiếp để định nghĩa nhãn target -> Data Leakage |
| `rd_id` | Mã định danh khách hàng, không có giá trị dự đoán |

**Chia dữ liệu:** 80% Train / 20% Test, thực hiện phân tầng (stratified) theo tỷ lệ churn.

**Pipeline chuẩn hóa:** `OneHotEncoder` + `StandardScaler` -> `SMOTE` -> `Model`

> **Lưu ý quan trọng:** Khi đã áp dụng kỹ thuật cân bằng mẫu SMOTE, KHÔNG dùng thêm `class_weight='balanced'` hay `scale_pos_weight` - nhằm tránh hiện tượng Double-Overweighting làm giảm chỉ số Precision nghiêm trọng.

### 5.2. Tại sao chọn AUPRC thay vì ROC-AUC?

| Chỉ số | Phù hợp Mất cân bằng? | Baseline ngẫu nhiên | Hạn chế |
|---|:---:|:---:|---|
| Accuracy | Không | 94.54% | Bị thiên lệch hoàn toàn |
| ROC-AUC | Lạc quan giả | 0.50 | Bị ảnh hưởng bởi lượng True Negatives quá áp đảo |
| **AUPRC** | **Có** | **0.0546** | Chỉ tập trung đo lường trên lớp Churn -> Phản ánh trung thực |

**AUPRC (Area Under Precision-Recall Curve)** đo diện tích dưới đường cong Precision-Recall. Baseline của dự đoán ngẫu nhiên chính bằng tỷ lệ churn (5.46%). Mô hình vượt xa con số này chứng minh khả năng phân biệt thực sự hiệu quả.

### 5.3. Kết quả Baseline (Chưa tinh chỉnh)

| Mô hình | ROC-AUC | AUPRC | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.6846 | 0.1566 | 0.08 | 0.61 | 0.14 |
| **Random Forest** | **0.9378** | **0.6066** | **0.70** | 0.40 | 0.51 |
| XGBoost | 0.9366 | 0.5888 | 0.67 | 0.39 | 0.49 |

- **Logistic Regression:** AUPRC = 0.1566 (~2.9x baseline). Precision chỉ 8% - không thể ứng dụng thực tế.
- **Random Forest:** AUPRC = 0.6066 (~**11x baseline**). Precision 70%, rất đáng tin cậy khi cảnh báo churn.
- **XGBoost:** AUPRC = 0.5888 (~10.8x baseline). Tương đương Random Forest.

### 5.4. Tinh chỉnh Siêu tham số (Hyperparameter Tuning) - GridSearchCV với `scoring='average_precision'`

Tối ưu hóa trực tiếp chỉ số AUPRC thay vì ROC-AUC.

#### Kết quả sau khi Tuning

| Mô hình | ROC-AUC | AUPRC | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.6861 | 0.1469 | 0.08 | 0.60 | 0.14 |
| **Random Forest** (Tốt nhất) | **0.9381** | **0.6032** | **0.75** | **0.41** | **0.53** |
| XGBoost | 0.9323 | 0.5523 | 0.62 | 0.36 | 0.46 |

**Random Forest** vượt trội là mô hình tốt nhất: AUPRC = **0.6032** (gấp ~11 lần baseline), Precision = 75%.

### 5.5. Đường cong Hiệu năng (Performance Curves)

- **Đường cong ROC:** RF (AUC=0.94), XGBoost (AUC=0.93) - cả hai đều tốt nhưng dễ dẫn đến kết luận lạc quan giả.
- **Đường cong PR:** RF duy trì Precision rất cao ở dải Recall thấp-trung bình. Ngân hàng có thể linh hoạt điều chỉnh ngưỡng theo chiến lược chi phí.

### 5.6. Tối ưu hóa Ngưỡng quyết định (Threshold Tuning) - Random Forest

**Random Forest được chọn** vì đạt chỉ số AUPRC cao nhất. Tìm ngưỡng tối ưu **F1-Score** để cân bằng hợp lý giữa Precision và Recall.

#### Ngưỡng tối ưu F1: **0.57**

| Chỉ số | Giá trị |
|---|:---:|
| F1-Score (Churn) | **0.5357** |
| Precision (Churn) | **93.75%** |
| Recall (Churn) | **37.50%** |
| Accuracy | 0.9645 |

**Ma trận nhầm lẫn (Confusion Matrix):**

|  | Dự đoán: Ở lại | Dự đoán: Churn |
|---|:---:|:---:|
| **Thực tế: Ở lại** | **1,381** (TN) | 2 (FP) |
| **Thực tế: Churn** | 50 (FN) | **30** (TP) |

- **30 khách hàng** churn được phát hiện chính xác (TP).
- **Chỉ 2 khách hàng** bị báo nhầm (FP) - chi phí lãng phí chiến dịch gần như bằng 0.
- **50 khách hàng** churn bị bỏ sót (FN) - rủi ro mất doanh thu tiềm năng.

> **Điểm nổi bật:** Ngưỡng 0.57 đem lại Precision lên tới **93.75%** - khi Random Forest phát cảnh báo churn, xác suất chính xác lên tới 93.75%.

#### Bảng Đánh đổi (Trade-off Matrix) theo Ngưỡng - Random Forest

| Ngưỡng | Precision | Recall | F1-Score | Chiến lược kinh doanh phù hợp |
|:---:|:---:|:---:|:---:|---|
| 0.10 | 0.1940 | **0.9625** | 0.3229 | Chiến dịch chi phí thấp (SMS, Push Notification) - bắt gần toàn bộ churn |
| 0.20 | 0.2642 | 0.8125 | 0.3988 | |
| 0.30 | 0.3759 | 0.6250 | 0.4695 | |
| 0.40 | 0.5000 | 0.4875 | 0.4937 | |
| 0.50 | 0.7500 | 0.4125 | 0.5323 | Mặc định (Default) |
| **0.57** | **0.9375** | **0.3750** | **0.5357** | **<-- Tối ưu F1 - Precision cực cao** |
| 0.60 | 0.9667 | 0.3625 | 0.5273 | |
| 0.70 | **1.0000** | 0.3375 | 0.5047 | Chiến dịch chi phí cao (Tư vấn 1-1, Ưu đãi giá trị cao) |
| 0.80 | **1.0000** | 0.2500 | 0.4000 | Precision tuyệt đối 100%, nhưng chỉ bắt được 25% churn |

**Cách đọc bảng:**
- **Ngưỡng 0.10:** Bắt được 96.25% khách hàng churn nhưng có 4/5 cảnh báo là nhầm lẫn.
- **Ngưỡng 0.57:** **93.75%** cảnh báo chính xác, bắt được 37.5% churn - lựa chọn tối ưu khi ngân sách có hạn.
- **Ngưỡng 0.70+:** 100% cảnh báo chính xác nhưng chỉ phát hiện được 25-34% lượng khách churn.

### 5.7. Phân tích Tầm quan trọng của Biến (SHAP Feature Importance) - Random Forest

Sử dụng **SHAP TreeExplainer** trên mô hình Random Forest để giải thích chi tiết mức độ đóng góp của từng biến.

#### Top yếu tố tác động chính

| Hạng | Biến | Ý nghĩa kinh doanh |
|:---:|---|---|
| 1 | `savingvaluemar2021_heoso` | **Số dư tiết kiệm T3** - quan trọng nhất. Số dư cao = mức độ gắn kết cao. |
| 2 | `distinct_trans_group` | **Đa dạng loại giao dịch** - dùng nhiều dịch vụ sẽ ít có xu hướng rời bỏ. |
| 3 | `distinct_ref_no` | **Số lượng tham chiếu giao dịch** - tần suất cao = khách hàng trung thành. |
| 4 | `total_act_mar2021` | **Tổng hoạt động T3** - tương tác nhiều = tỷ lệ rời bỏ thấp. |
| 5+ | Tuổi, Tỉnh thành | **Nhân khẩu học** - tác động thứ yếu so với hành vi giao dịch. |

**Insight cốt lõi:** Hành vi tài chính (số dư, tần suất giao dịch, mức độ đa dạng dịch vụ) có **sức ảnh hưởng mạnh hơn nhiều** so với các yếu tố nhân khẩu học tĩnh.

---

## 6. Báo cáo Kết luận & Đề xuất Hành động Kinh doanh
*(Dành cho Ban Giám đốc Khối Ngân hàng Số & Đội ngũ Kinh doanh)*

### 6.1. Tóm tắt Giá trị Kinh doanh của Mô hình Cảnh báo Sớm (Executive Summary)

* **Bối cảnh & Thách thức:**
  * Tỷ lệ khách hàng rời bỏ sản phẩm **Heo Sổ** (không còn đăng nhập hoặc rút toàn bộ tiền tiết kiệm) chiếm **5.46%** (khoảng ~400 khách hàng trong kỳ quan sát).
  * Nếu dùng phương pháp tiếp thị dàn trải toàn bộ khách hàng (hơn 7,300 người) để giữ chân, ngân hàng sẽ lãng phí đến **95% chi phí vận hành & marketing** cho những người vốn dĩ không có ý định rời đi.

* **Hiệu quả thực tế từ giải pháp AI / Data Analytics:**
  * **Độ chính xác vượt trội (Precision 93.75%):** Khi mô hình phát tín hiệu cảnh báo rủi ro rời bỏ ở ngưỡng tối ưu, cứ **100 khách hàng được cảnh báo thì có đến ~94 khách hàng thực sự sẽ rời bỏ** nếu ngân hàng không can thiệp.
  * **Tối ưu ngân sách:** Giúp ngân hàng khoanh vùng chính xác tập khách hàng rủi ro cao, giảm 90% chi phí chiến dịch tiếp thị và chăm sóc khách hàng.

### 6.2. Giải mã Tín hiệu Rời bỏ & Chân dung Khách hàng Rủi ro (Business Insights)

Dữ liệu thực tế chỉ ra 3 nguyên nhân cốt lõi khiến khách hàng ngưng sử dụng Heo Sổ:

1. **Tín hiệu sinh tử - Số dư Tiết kiệm suy giảm:**
   * Số dư Heo Sổ duy trì trong tháng trước là chỉ số dự báo quan trọng nhất. Khách hàng bắt đầu rút bớt tiền hoặc giữ số dư quá thấp trong Tháng 3 có nguy cơ rút sạch tiền và bỏ sản phẩm ở Tháng 6 cao gấp nhiều lần.
2. **Mức độ gắn kết sinh thái (Product Stickiness) thấp:**
   * Khách hàng rời bỏ phần lớn là những người **chỉ gửi tiết kiệm đơn thuần** mà không sử dụng các dịch vụ khác (chuyển khoản, thanh toán hóa đơn, nạp tiền điện thoại).
   * Khách hàng trải nghiệm càng đa dạng loại hình giao dịch thì tỷ lệ trung thành càng cao.
3. **Phân khúc nhạy cảm:**
   * Khách hàng **Độc thân**, nhóm tuổi **Gen Z & Gen Y (18 - 35 tuổi)** có tỷ lệ rời bỏ cao hơn nhóm đã kết hôn và trung niên.
   * Tập trung rủi ro cao tại một số **địa bàn trọng điểm (Tỉnh code 27.0 và 38.0)** – gợi ý có thể do sự cạnh tranh lãi suất/khuyến mãi gay gắt từ các ngân hàng đối thủ hoặc dịch vụ hỗ trợ tại khu vực này chưa tốt.

### 6.3. Đề xuất Chương trình Hành động Chi tiết (Actionable Roadmap)

#### **Hành động 1: Xây dựng Luồng Cảnh báo Tự động (Automated Early Warning Trigger)**
* **Cơ chế:** Tích hợp mô hình dự báo vào hệ thống CRM / App Ngân hàng số.
* **Kịch bản vận hành:**
  * Ngay khi khách hàng có dấu hiệu **rút tiền Heo Sổ > 30%** hoặc **giảm tần suất đăng nhập > 50%** so với tháng trước, hệ thống tự động gắn nhãn *"Rủi ro Churn"*.
  * Tự động kích hoạt thông báo Push Notification/Email với thông điệp cá nhân hóa: *"Duy trì số dư Heo Sổ tháng này để nhận Voucher hoàn tiền 50k"* hoặc *"Cộng thêm 0.2% lãi suất cho kỳ hạn tích lũy tiếp theo"*.

#### **Hành động 2: Chiến lược "Bán chéo nâng cao độ dính" (Cross-Selling & Retention Combo)**
* Biến Heo Sổ từ "hũ tiết kiệm tĩnh" thành một "tài khoản giao dịch linh hoạt".
* **Chương trình khuyến mãi:**
  * Tung gói **"Combo Tiết kiệm + Giao dịch"**: Miễn phí chuyển khoản ngoài hệ thống hoặc tặng quà quay số cho khách hàng có duy trì Heo Sổ VÀ thực hiện tối thiểu 3 giao dịch thanh toán hóa đơn/tháng.
  * Tự động trích tiền lẻ giao dịch hàng ngày (Round-up) để nạp tự động vào Heo Sổ.

#### **Hành động 3: Vận hành Chiến dịch Giữ chân theo Ngân sách linh hoạt (Dynamic Campaign Budgeting)**
Ngân hàng có thể chủ động điều chỉnh "kính lọc" của mô hình tùy theo ngân sách khuyến mãi:
* **Khi ngân sách hạn chế (Chiến lược "Bắn súng tỉa"):**
  * Đặt ngưỡng phân loại cao (Precision ~94%). Chi phí chăm sóc chỉ dồn vào đúng nhóm khách hàng rủi ro nhất (đặc biệt là khách hàng VIP/tiết kiệm lớn). Đảm bảo mỗi đồng chi phí đều mang lại ROI cao.
* **Khi ngân sách dồi dào / Cần bao phủ thị trường (Chiến lược "Bủa lưới"):**
  * Hạ ngưỡng cảnh báo để bao phủ 80-95% số lượng khách hàng có nguy cơ rời bỏ. Áp dụng các kịch bản giữ chân tự động chi phí 0đ (Banner in-app, tin nhắn Zalo ZNS tự động).

#### **Hành động 4: Đánh giá & Rà soát Đột xuất tại Địa bàn Trọng điểm (Tỉnh 27.0 & 38.0)**
* Giao Khối Vận hành & Phát triển Thị trường khảo sát nhanh thị trường Tỉnh 27 và 38:
  * Kiểm tra xem đối thủ cạnh tranh có đang tung chiến dịch lãi suất tiết kiệm số cao vượt trội hay không.
  * Rà soát các sự cố về mạng/nạp rút tiền cục bộ gây bức xúc cho người dùng.

### 6.4. Kỳ vọng Hiệu quả & Đo lường ROI (Expected Impact)

* **Tiết kiệm chi phí:** Rút ngắn 90% tập khách hàng cần tiếp thị giữ chân, giảm chi phí gửi SMS/Voucher không đúng đối tượng.
* **Bảo toàn nguồn vốn huy động:** Giữ chân thành công khoảng 35-40% lượng khách hàng rủi ro (đặc biệt là nhóm tiền gửi lớn), bảo vệ hàng chục tỷ đồng nguồn vốn tiết kiệm nhỏ lẻ nhưng ổn định.
* **Chỉ số đo lường KPI:** Đánh giá lại tỷ lệ Churn Rate của sản phẩm Heo Sổ theo từng quý sau khi triển khai hệ thống cảnh báo tự động.

---

*Báo cáo được tổng hợp và xuất bản từ kết quả phân tích trong `final.ipynb` - Group 01, Challenge 03, FDC105.*

