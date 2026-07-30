## 📓 CREEDIT RISK MODELING

Dự án xây dựng MÔ HÌNH HÓA RỦI RO TÍN DỤNG sử dụng tập dữ liệu từ Home Credit Default Risk. Dự án này thực hiện mô phỏng theo dự án tại một ngân hàng thực bao cồm phần Xác xuất rủi ro vỡ nợ được mô hình hóa, đánh giá khả năng phân biệt, khả năng hiệu chỉnh (calibration) và tính ổn định qua những chỉ số đánh giá của ngành và đánh giá lại mô hình dưới các dạng phân bố khác với phân bố gốc.

Pipeline được chia thành 3 notebook tương ứng với 3 giai đoạn của quy trình xây dựng PD Scorecard: **Feature Engineering → Fine Classing/WOE → Model Development & Validation**. Mỗi notebook nhận input từ notebook trước và xuất ra dataset cho notebook sau.

| # | Notebook | Giai đoạn | Input | Output |
|---|----------|-----------|-------|--------|
| 1 | `1_feature_engineering.ipynb` | Feature Engineering | 6 bảng gốc (application, bureau, bureau_balance, POS_CASH, credit_card_balance, installments, previous_application) | `all_feature_{train/val/test}.csv` (~216 features) |
| 2 | `2_FineClassing_ManualReview.ipynb` | Fine Classing / WOE / Manual Review | `all_feature_{train/val/test}.csv` | `woe_logistics_{train/val/test}.csv`, `bins_feature.csv` (44 features) |
| 3 | `3_ModelDevelopment.ipynb` | Model Development & Validation | WOE datasets + bins | Logistic Regression Scorecard + validation report (37 features) |

---

### 01 — Feature Engineering

**Mục tiêu:** chuyển 6 bảng dữ liệu thô của Home Credit (mỗi bảng ở một cấp độ granularity khác nhau — hồ sơ vay, lịch sử tín dụng ngoài, lịch sử vay/trả góp nội bộ) thành một bảng feature duy nhất ở mức khách hàng (`SK_ID_CURR`), sẵn sàng cho bước variable selection.

**Các bước chính:**
- **Application (hồ sơ hiện tại):** xử lý bất thường của `DAYS_EMPLOYED` (365243), tạo các tỷ số năng lực trả nợ (annuity/income, credit/income, credit/goods), tuổi, số năm đi làm, và các thống kê tổng hợp từ 3 external score (mean/min/max/std/missing count).
- **Bureau + Bureau_balance:** đây là cấu trúc 2 tầng (mỗi `SK_ID_BUREAU` có nhiều tháng lịch sử trong bureau_balance) → phải aggregate bureau_balance theo window (6/12/24 tháng, DPD, bad ratio) trước, rồi mới join ngược lên bureau và aggregate lần 2 lên `SK_ID_CURR` (exposure, số khoản active/closed, mức độ delinquency theo từng window).
- **POS_CASH_balance:** aggregate lịch sử vay trả góp/tiền mặt nội bộ theo toàn bộ lịch sử và 12 tháng gần nhất, cộng thêm snapshot gần nhất để đo gánh nặng trả nợ hiện tại (future installments).
- **Credit card balance:** tính utilization, xu hướng utilization (3 tháng gần vs 3 tháng trước), DPD, payment ratio ở statement gần nhất.
- **Installments payments:** tính payment ratio, tỷ lệ trả thiếu, số ngày trễ hạn (toàn bộ lịch sử và 12 tháng gần nhất).
- **Previous_application:** tỷ lệ duyệt/từ chối, số hồ sơ gần đây (12m), đặc điểm của hồ sơ được duyệt gần nhất.
- Toàn bộ feature engineering được viết thành hàm tái sử dụng (`*_features()`) và áp dụng đồng nhất cho cả train/validation/test để tránh leakage.

**Lưu ý kỹ thuật:** dataset không có cột ngày tuyệt đối nên split được thực hiện theo stratified random 70/15/15 trên tập TRAIN (vì tập test của HOME CREDIT không có cột TARGET là PD hay không) thay vì out-of-time — đây là giới hạn, khi chia dữ liệu như vậy, ngẫu nhiên model sẽ học được các tính chất phân phối trong cùng 1 tập. Ví dụ nếu có tập OOT, ta có thể Train model từ năm 2015-2018, Validation trên 2019-2020 và Test trên 2021-2022. Nếu không có OOT, ta buộc phải gộp từ 2015-2022 và vô tình việc chia ngẫu nhiên đã giúp model học được 1 phần data từ 2019-2022, khi đó các kiểm định chỉ số về sau của Validaiton và Test có thể sẽ "đep" hơn bình thường.

---

### 02 — Fine Classing, WOE & Manual Review

**Mục tiêu:** từ ~216 feature thô, chọn lọc xuống một tập biến nhỏ, có ý nghĩa nghiệp vụ, đơn điệu và không đa cộng tuyến — vì input của Logistic Regression Scorecard cần tính tuyến tính và diễn giải được.

**Pipeline chọn biến (funnel):**

| Bước | Mô tả | Số feature còn lại |
|------|-------|---------------------|
| Feature profiling | Loại near-zero-variance, missing quá cao (>80%), dominant value quá cao (>90%) | 216 → 187 |
| Fine Classing + IV filter | Chia bin theo quantile (numeric) / giữ nguyên nhóm (categorical), tính Information Value, loại nhóm có chỉ số IV được coi là "Very Weak" | 187 → 100 |
| Correlation Filter | Loại biến tương quan cao (threshold 0.7) để tránh đa cộng tuyến trước khi vào model | 100 → 50 |
| Manual Review (business logic) | Ép hướng đơn điệu tăng/giảm theo nghiệp vụ cho các biến rõ ràng; loại thêm 5 biến do missing cao/không có ý nghĩa | 50 → 44 |

**Note:** ở bước Manual Review, thay vì ép tất cả biến phải đơn điệu, các biến khó xác định hướng được giữ nguyên cấu trúc bin gốc — vì ép đơn điệu sai sẽ làm model học sai bản chất rủi ro, dù việc này có thể khiến model khó diễn giải và dễ overfit hơn. Tôi sẽ cải thiện các biến không có xu hưỡng rõ ràng bằng việc kết hợp các bin khoa học hơn về sau

**Output:** WOE-transformed dataset (44 features) cho train/validation/test + bảng bin definition (`bins_feature.csv`) dùng để map WOE cho dữ liệu mới.

---

### 03 — Model Development & Validation

**Mục tiêu:** fit Logistic Regression trên WOE data, kiểm tra tính ổn định của hệ số, loại bỏ biến không có ý nghĩa thống kê, chuyển đổi thành scorecard theo chuẩn ngân hàng (PDO/Factor/Offset), và validate model trên 3 tập dữ liệu.

**Các bước chính:**
- **Baseline model:** fit Logit trên 44 biến WOE, kiểm tra 3 điều kiện: hệ số ổn định (coefficient/SE threshold), đa cộng tuyến (VIF < 5), và ý nghĩa thống kê (p-value < 0.05).
- **Loại biến không có ý nghĩa:** 7 biến bị loại do không qua được kiểm định thống kê → còn lại 37 biến cho model cuối cùng.
- **Scorecard construction:** scale hệ số logistic thành điểm số theo chuẩn ngành: `Base Score = 600, PDO = 20, PD target = 2%`.
- **Cutoff analysis:** dựng bảng approval rate / bad rate theo từng ngưỡng điểm, xác định điểm gãy (breakpoint) nơi bad rate không còn ổn định do mẫu quá nhỏ.
- **Model Validation:** đánh giá discriminatory power (Gini, KS) trên cả 3 tập train/validation/test, kiểm tra decile bad rate có giảm dần đơn điệu (monotonic) hay không.
- **Population Shift Sensitivity Analysis:** kiểm tra xem model có chịu được danh mục trong tương lai khi các khách hàng rủi ro cao tăng lên hay không. Tạo ra các kịch bản có thể xảy ra trong tương lai


**Output:** model logistic cuối cùng (37 biến) + bảng điểm scorecard + báo cáo validation.

---

## 📊 Kết quả chính (Key Results)

**Funnel chọn biến:**

`216 features thô → 187 (profiling) → 100 (IV filter) → 50 (correlation filter) → 44 (manual review) → 37 (significance filter, model cuối)`

**Discriminatory power (Gini / KS):**

| Dataset | Gini | KS |
|---------|------|-----|
| Train | 0.522 | 0.391 |
| Validation | 0.521 | 0.388 |
| Test | 0.520 | 0.395 |

→ Gini ổn định quanh 0.52 trên cả 3 tập, không có dấu hiệu overfitting rõ rệt (chênh lệch train/test < 0.01). Và như đã nói ở phần "Lưu ý kỹ thuật" của Feature Engineering, các kết quả này vô tình không còn nhiều đáng tin do việc chia dữ liệu

**Scorecard scaling:**
- Base Score = 600 tại PD = 2%, PDO (Points to Double the Odds) = 20.
- Cutoff phân tích cho thấy tại điểm ~880, bad rate quan sát không còn ổn định về mặt thống kê do số lượng khách hàng đạt ngưỡng này quá nhỏ (43 hồ sơ) 

**Model behavior:**
- Bad rate giảm đơn điệu qua từng decile điểm số (cả ở tập Train và tập Test), đúng như kỳ vọng của một scorecard hoạt động tốt.
