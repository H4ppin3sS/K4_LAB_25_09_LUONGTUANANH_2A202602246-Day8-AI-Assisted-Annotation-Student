# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Lương Tuấn Anh

Công cụ gán nhãn đã dùng:  CVAT cá nhân

Sao chép file này thành `reports/REPORT.md` rồi  vào các chỗ . Mọi con số phải truy được
từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round*.json` hoặc
`outputs/round*_diff.md`. Không coi nhãn test do mô hình tạo là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng đệm ở giữa, để hạn chế việc các frame rất gần nhau về thời gian xuất hiện ở cả hai tập. Các frame gần nhau thường có bối cảnh và đối tượng tương tự, nên nếu chia ngẫu nhiên thì mô hình có thể gặp các ảnh rất giống nhau trong quá trình chọn/gán nhãn và trong tập kiểm thử.

Khi đó, số đo trên test set có thể bị lệch theo hướng quá lạc quan vì test không còn hoàn toàn độc lập về mặt thời gian. Cách chia theo thời gian và có vùng đệm giúp giảm nguy cơ thông tin hình ảnh gần trùng giữa pool và test làm cho kết quả đánh giá cao hơn khả năng tổng quát thực tế.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

Tập kiểm thử gồm 20 ảnh với 403 box tham chiếu; bỏ qua 14 box có chiều cao dưới 16 px. Ngưỡng IoU là 0.5 và P, R, F1 được tính tại confidence 0.25.

Theo số đo theo kích thước, recall của xe nhỏ chỉ là 0.182, thấp hơn rõ rệt so với xe trung bình (0.547) và xe lớn (0.561). Điều này cho thấy mô hình khởi đầu lạnh gặp khó khăn hơn với các đối tượng nhỏ.

Một trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai là những xe quá xa hoặc quá mờ. Trong blind_scan, khi quan sát độc lập frame 0099, có 23 xe nhìn thấy bằng mắt và hai vị trí dễ bị AI bỏ sót hoặc vẽ sai là xe ở quá xa/không thể xác định hoặc quá mờ. Vì vậy, những trường hợp này cần được kiểm tra lại chất lượng và khả năng quan sát của nhãn tham chiếu trước khi quy lỗi hoàn toàn cho mô hình.

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ----- | ----- | ---------- | --------- | ---- | --------------------- | ------ | ------ | -- | ------- | -------- | ------- |

| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | <br /><br /><br />0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| - | --------------------------------------- | - | - | ----- | -- | ----------------------- | ----- | ----- | ----- | ----- | ----- |

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

điểm ưu tiên của một frame được tạo từ ba thành phần: độ bất định U, mức độ khó/độ hữu ích của mẫu A, và một thành phần liên quan đến độ đa dạng hoặc khác biệt D. Các trọng số W_U, W_A, W_D quyết định mức đóng góp của từng thành phần vào điểm cuối. Các vòng học chủ động (active learning)

MIN_GAP_S: dùng để tạo khoảng cách tối thiểu theo thời gian giữa các frame được chọn. Vai trò của nó là hạn chế việc chọn quá nhiều ảnh gần trùng nhau, từ đó dành ngân sách rà nhãn cho các thời điểm khác nhau.

Ba frame được sử dụng trong lựa chọn là:

* frame_0182: frame được ưu tiên trong selection; trong diff vòng 1 có 13 box pre-label, 12 box được giữ, 1 box bị xoá và 11 box mới được thêm, tạo thành 23 box cuối.
* frame_0331: có 20 box pre-label, 16 box được giữ, 1 box được chỉnh, 3 box bị xoá và 7 box được thêm, tạo thành 24 box cuối.
* frame_0392: có 13 box pre-label, 12 box được giữ, 1 box được chỉnh và 7 box được thêm, tạo thành 20 box cuối.
  Một frame khác là frame_0099. Đây là frame được dùng cho blind scan. Quan sát độc lập trước khi xem pre-label ghi nhận 23 xe nhìn thấy bằng mắt. Sau khi đối chiếu pre-label, diff cho thấy 13 box ban đầu, 10 box được giữ, 3 box được chỉnh và 8 box được thêm, không có box bị xoá, tạo thành 21 box cuối.
  Điểm bất định không chứng minh rằng một ảnh chắc chắn sẽ cải thiện mô hình. Nó chỉ là tín hiệu để ưu tiên ảnh có khả năng chứa thông tin hữu ích hoặc trường hợp khó. Muốn chứng minh ảnh đó thực sự cải thiện mô hình cần gán nhãn đúng, huấn luyện và đánh giá lại trên cùng tập kiểm thử.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.


| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ----- | ----- | ---------- | --------- | ---- | --------------------- | ------ | ------ | -- | ------- | -------- | ------- |

| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| - | --------------------------------------- | - | - | ----- | -- | ----- | ----- | ----- | ----- | ----- | ----- |

| 1 | yolov8n fine-tune vong 1..1 | 12 | 265 | 0.367 | -0.405 | 1.000 | 0.015 | 0.029 | 0.000 | 0.007 | 0.098 |
| - | --------------------------- | -- | --- | ----- | ------ | ----- | ----- | ----- | ----- | ----- | ----- |

Tập test được giữ cố định gồm 20 ảnh và 403 box tham chiếu, với 14 box nhỏ hơn 16 px được bỏ qua.

Ở vòng 1, có 151 box được giữ nguyên, 8 box được chỉnh sửa, 10 box bị xoá và 106 box được thêm mới. Tổng số pre-label box là 169 và số box cuối là 265; accept rate là 0.8935.

Một số ví dụ cụ thể từ diff và review log:

* frame_0099: 10 box được giữ, 3 box chỉnh sửa, 8 box thêm mới; không có box bị xoá. Review log ghi nhận đã thêm một xe tối gần mép trái vì nhìn thấy thân xe và đủ ranh giới theo guideline.
* frame_0107 : 12 box được giữ, 1 box bị xoá và 12 box được thêm. Review log ghi nhận trường hợp hai xe sát nhau làm pre-label tạo box chung; box chung được xoá và hai xe được thêm thành hai box riêng vì nhìn rõ cả hai xe.
* frame_0182: 12 box được giữ, 1 box bị xoá và 11 box được thêm. Review log ghi nhận một box bị vẽ vượt ra khỏi xe và đã được chỉnh vì xe nhìn rõ

So với cold start, AP50 giảm từ 0.771 xuống 0.367, tức giảm 0.405. P tăng từ 0.925 lên 1.000 nhưng recall giảm rất mạnh từ 0.489 xuống 0.015, làm F1 giảm từ 0.640 xuống 0.029. Recall của cả ba nhóm kích thước cũng giảm: small từ 0.182 xuống 0.000, medium từ 0.547 xuống 0.007 và large từ 0.561 xuống 0.098.

Vì AP50 và recall đều giảm mạnh sau fine-tune, kết quả này cần được kiểm tra trước khi tiếp tục train thêm. Một khả năng cần xem xét là dữ liệu train chỉ gồm 12 ảnh nhưng đã có 265 box, trong khi test có 20 ảnh và 403 box tham chiếu. Tuy nhiên, không nên kết luận nguyên nhân chỉ từ các số liệu này

Một ca khó là xe quá xa hoặc quá mờ. Đây là trường hợp cần thận trọng vì người rà cũng khó xác định chính xác vị trí và ranh giới xe. Do đó, không nên tự động xem mọi box bị bỏ sót trong những vùng này là lỗi mô hình.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

Ở vòng 1, kết quả sau fine-tune kém hơn cold start trên tập test được báo cáo. AP50 giảm từ 0.771 xuống 0.367; recall giảm từ 0.489 xuống 0.015 và F1 giảm từ 0.640 xuống 0.029. Recall theo kích thước cũng giảm mạnh, đặc biệt recall xe nhỏ giảm từ 0.182 xuống 0.000.

Với kết quả này, trước khi train thêm tôi sẽ kiểm tra lại dữ liệu train, các thay đổi nhãn và pipeline fine-tune thay vì chỉ tăng thêm số vòng huấn luyện. Cụ thể, cần kiểm tra xem các box được thêm/sửa/xoá có đúng guideline hay không, dữ liệu train có được đóng gói đúng hay không, và kết quả sau fine-tune có đang gặp vấn đề về phân bố hoặc cấu hình huấn luyện hay không.

Hai nhóm ca có thể ưu tiên rà ở vòng sau là:

1. Các xe nhỏ hoặc ở xa/khó nhìn, vì recall của nhóm small ở vòng 1 là 0.000 và blind scan cũng chỉ ra các trường hợp xe quá xa hoặc quá mờ là vùng dễ bỏ sót.
2. Các frame có nhiều đối tượng sát nhau hoặc pre-label có xu hướng gộp box, như trường hợp frame_0107, nơi hai xe sát nhau dẫn đến một box chung và cần xoá box cũ rồi thêm hai box riêng.

Chi phí rà nhãn cần cân nhắc vì vòng 1 đã có 106 box được thêm mới, 8 box chỉnh sửa và 10 box bị xoá trên 12 ảnh. Việc chọn thêm các frame gần nhau cũng có nguy cơ lãng phí công rà vì chúng có thể chứa cảnh tương tự; do đó cần tiếp tục dùng khoảng cách tối thiểu giữa các frame khi chọn mẫu.

Tập kiểm thử chỉ có 20 ảnh và 403 box tham chiếu, đồng thời bỏ qua 14 box có chiều cao dưới 16 px. Vì vậy, các số đo có thể chưa phản ánh đầy đủ các trường hợp xe rất nhỏ. Ngoài ra, nhãn tham chiếu do mô hình tạo chưa được rà thủ công hoàn toàn, nên không nên coi mọi khác biệt giữa dự đoán và nhãn tham chiếu là lỗi của mô hình.

Nếu AP50 giảm, sẽ kiểm tra trước chất lượng và cấu trúc nhãn train, các box bị sửa/xoá/thêm, việc đóng gói dữ liệu, cấu hình fine-tune và sự nhất quán giữa train/test. Sau đó mới quyết định có nên train thêm hay thay đổi chiến lược chọn mẫu.
