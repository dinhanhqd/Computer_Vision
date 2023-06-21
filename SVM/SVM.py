# import the necessary packages
# svm để triển khai thuật toán SVM được cung cấp thư viện scikit-learn
from sklearn import svm
# LabelEncoder, một tiện ích để chuyển nhãn được biểu thị dưới dạng chuỗi thành số nguyên
from sklearn.preprocessing import LabelEncoder
# Hàm train_test_split được sử dụng để tách dữ liệu: dứ liệu train và dữ liệu test.
from sklearn.model_selection import train_test_split
# Hàm classification_report để đánh giá hiệu suất của bộ phân loại và in bảng kết quả
from sklearn.metrics import classification_report
# Gọi Bộ tiền xử lý dữ liệu (để chuyển kích thước ảnh về 32 x 32)
from preprocessing import SimplePreprocessor
# Gọi bộ nạp dữ liệu, tham số verbose=500 cho biết nạp 500 ảnh để xử lý 1 lần
from datasets import SimpleDatasetLoader
from imutils import paths
import pickle

# Lấy danh sách các ảnh trong folder
print("[INFO] Nạp ảnh...")
imagePaths = list(paths.list_images("datasets/dataSkin"))
# Khởi tạo bộ tiền xử lý ảnh và nạp ảnh từ folder, thay đổi kích thước ảnh
sp = SimplePreprocessor(32, 32) # Thiết lập kích thước ảnh 32 x 32

sdl = SimpleDatasetLoader(preprocessors=[sp]) # Tạo bộ nạp dữ liệu
(data, labels) = sdl.load(imagePaths, verbose=500)#Nạp dữ liệu ảnh (mỗi lần 500 ảnh) đã được gắn nhãn trong "datasets/animals"
data = data.reshape((data.shape[0], 3072))# biểu diễn lại mảng dữ liệu ảnh

# Hiển thị thông tin bộ nhớ đã dùng
# Dung lượng = (32x32x3) *3000 ảnh = 3072*3000 ~ 9MB
print("[INFO] Dung lượng bộ nhớ chứa dữ liệu ảnh: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# Chuyển nhãn từ chuỗi sang số
le = LabelEncoder()
labels = le.fit_transform(labels)

# Chia dữ liều thành 2 phần: 75% để train và 25% để test
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)

print("[INFO] Đánh giá Bộ phân lớp SVM ...")

# Tạo model (Bộ phân loại SVM với kernel là Radial Basic Function)
model = svm.SVC(kernel='rbf', gamma='scale')
# SVC viết tắt Support Vector Classification
# Tham khảo thêm tại https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# Huấn luyện model (Bộ phân loại SVM)
model.fit(trainX, trainY)

#Lưu file model (Bộ phân loại SVM sau khi được huấn luyện)
pickle.dump(model, open("SVM.model", 'wb'))

# Đánh giá hiệu suất và hiển thị kết quả
print(classification_report(testY, model.predict(testX),target_names=le.classes_))

