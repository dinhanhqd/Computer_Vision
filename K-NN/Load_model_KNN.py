# import các packages cần thiết
from preprocessing import SimplePreprocessor # Import modul SimplePreprocessor
from datasets import simpledatasetloader # Import modul simpledatasetloader
import cv2
import pickle  #Thư viện này để đọc file model

# Khởi tạo danh sách nhãn
classLabels = ["cat", "dog", "panda"]

print("[INFO] Đang nạp ảnh để bộ phân lớp dự đoán...")

# Thiết lập kích thước ảnh 32 x 32
sp = SimplePreprocessor(32, 32)

# Tạo bộ nạp dữ liệu ảnh
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp])

# Nạp dữ liệu file ảnh và lưu dưới dạng mảng
data, _ = sdl.load(["Image\\h2.jpg"])

#Thay đổi cách biểu diễn mảng dữ liệu ảnh
data = data.reshape((data.shape[0], 3072))

# Nạp model KNN đã train
print("[INFO] Nạp model k-NN ...")
model = pickle.load(open('knn.model', 'rb'))

# Dự đoán
print("[INFO] Thực hiện dự đoán ảnh để phân lớp...")
preds = model.predict(data) # Trả về danh sách nhãn dự đoán: 0->cat, 1->dog, 2->Panda


#Đọc file ảnh
image = cv2.imread("Image\\h2.jpg")
# Viết label lên ảnh
cv2.putText(image, "label: {}".format(classLabels[preds[0]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# Hiển thị ảnh
cv2.imshow("Image", image)
cv2.waitKey(0)







