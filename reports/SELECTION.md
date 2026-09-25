# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box: 

1. frame_0182.jpg (0.9591, 72.8): Điểm cao nhất trong top 50, đồng thời có 28 box và 18 box bị đánh dấu ambiguous. Đây là trường hợp đáng ưu tiên để rà thủ công vì điểm tổng thể cao nhưng vẫn có nhiều vùng cần kiểm tra.
2. frame_0331.jpg (0.9154, 132.4): Điểm cao, có 47 box — số lượng box lớn trong nhóm đầu — và 18 box ambiguous, phù hợp để kiểm tra nhiều đối tượng và các trường hợp khó.
3. frame_0099.jpg (0.9063, 39.6): Điểm cao và nằm ở một thời điểm khác đáng kể so với các frame quanh 130–150 giây, giúp tránh chỉ tập trung kiểm tra một đoạn thời gian.
4. frame_0227.jpg (0.8915, 90.8): Điểm tương đối cao và bổ sung một mốc thời gian trung gian, với 37 box và 14 box ambiguous.
5. frame_0227.jpg (0.8874, 156.8): Điểm thấp hơn các frame trên nhưng vẫn thuộc nhóm đầu; có 35 box và 12 box ambiguous. Frame này giúp kiểm tra thêm một đoạn thời gian cuối của dữ liệu thay vì chỉ chọn các điểm có score cao nhất.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet: 

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:

1. frame_0182.jpg — rank 1, score 0.9591, 28 boxes, 18 ambiguous.
2. frame_0331.jpg — rank 5, score 0.9154, 47 boxes, 18 ambiguous.
3. frame_0392.jpg — rank 15, score 0.8874, 35 boxes, 12 ambiguous.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do: Một frame có điểm cao nhưng không chọn là frame_0369.jpg (rank 2, score 0.9324, 147.6 giây). Lý do không đưa vào năm frame cuối là nó nằm rất gần frame_0372.jpg (148.8 giây), nên việc chọn thêm nhiều frame trong cùng đoạn thời gian có thể làm giảm độ đa dạng của mẫu rà thủ công.

Điều phép chọn này chưa chứng minh về chất lượng mô hình: Phép chọn này chỉ cho biết những frame nào được ưu tiên rà thủ công dựa trên các chỉ số có trong selection_round1 như score, U, A, D, số lượng box và số lượng box ambiguous. Nó chưa chứng minh precision, recall, mAP hay chất lượng annotation thực tế của mô hình trên toàn bộ tập dữ liệu. Việc một frame có score cao cũng không đồng nghĩa mọi bounding box trong frame đó đều đúng; cần đối chiếu với ảnh và ground truth/annotation do người kiểm tra tạo ra mới có thể đánh giá chất lượng thực tế.
