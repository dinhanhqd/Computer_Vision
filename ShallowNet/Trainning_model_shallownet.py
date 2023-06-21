# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Cách dùng Trainning_model_shallownet.py -d datasets/animals

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="Nhập folder chứa tập dữ liệu")
args = vars(ap.parse_args(["-d", "datasets"]))
imagePaths = list(paths.list_images(args["dataset"]))

# Bước 1. Chuẩn bị dữ liệu
# Khởi tạo tiền xử lý ảnh
sp = SimplePreprocessor(32, 32) # Thiết lập kích thước ảnh 32 x 32
iap = ImageToArrayPreprocessor() # Gọi hàm để chuyển ảnh sang mảng

# Nạp dataset từ đĩa rồi co dãn mức xám của pixel trong vùng [0,1]
print("[INFO] Nạp ảnh...")

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Chia tách dữ liệu vào 02 tập, training: 75% và testing: 25%
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)
# Chuyển dữ liệu nhãn ở số nguyên vào biểu diễn dưới dạng vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Bước 2. Xây dựng và train model
# Tạo bộ tối ưu hóa cho mô hình
opt = SGD(learning_rate=0.005)

# Khởi tạo mô hình mạng, biên dịch mô hình
print("[INFO] Tạo mô hình...")
model = ShallowNet.build(width=32, height=32, depth=3, classes=5)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

print("[INFO] training mạng ...")
# Train model
# Muốn lấy dữ liệu Validation từ Tập trainning thì thêm tham số:
# validation_split = giá trị trong đoạn 0 đến 1
H = model.fit(trainX, trainY, validation_split = 0.1, batch_size=32, epochs=30, verbose=1)

# Nếu muốn lấy dữ liệu từ tập Testing để làm Validation thì sử dụng tham số:
# validation_data=(testX, testY). Lúc đó dữ liệu này sẽ ghi chồng lên
# dữ liêu được tạo từ validation_split = giá trị trong đoạn 0 đến 1 (nếu có)

#H = model.fit(trainX, trainY, validation_split = 0.1,validation_data=(testX, testY),batch_size=32, epochs=100, verbose=1)
model.save("md1.h5")
model.summary() # Hiển thị tóm tắt các tham số của model

# Bước 3. Đánh giá mạng
print("[INFO] Đánh giá mạng...")
# Nạp phần Data của Tập Testing để đánh giá model (đánh giá dự đoán)
predictions = model.predict(testX, batch_size=32)
# In các tham số sự đoán, bao gồm Label của Data
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=["DaThuong", "NotRuoi", "BenhSoi","U_Lanh","UngThu"]))

# Vẽ kết quả trainning: mất mát (loss) và độ chính xác (accuracy) quá trình trainning
# Cách thứ nhất

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 30), H.history["loss"], label="Mất mát khi trainning")
plt.plot(np.arange(0, 30), H.history["val_loss"], label="Mất mát validation")
plt.plot(np.arange(0, 30), H.history["accuracy"], label="Độ chính xác khi trainning")
plt.plot(np.arange(0, 30), H.history["val_accuracy"], label="Độ chính xác validation ")
plt.title("Biểu đồ hiển thị mất mát và độ chính xác khi Training")
plt.xlabel("Epoch #")
plt.ylabel("Mất mát/Độ chính xác")
plt.legend()
plt.show()

# Cách thứ 2
# plot the training loss and accuracy
"""
def plot_trend_by_epoch(tr_value,val_value,title,y_plot,figure):
    epoch_num =range(len(tr_value))
    plt.plot(epoch_num,tr_value,'r')
    plt.plot(epoch_num, val_value,'b')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_plot)
    plt.legend(['Training ' + y_plot, 'Validation ' + y_plot])
    plt.savefig(figure)
tr_accuracy,val_accuracy = H.history["accuracy"],H.history["val_accuracy"]
plot_trend_by_epoch(tr_accuracy,val_accuracy,"Model Accuracy","Accuracy","plot_accu.png")
plt.clf()
tr_loss,val_loss = H.history["loss"],H.history["val_loss"]
plot_trend_by_epoch(tr_loss,val_loss,"Model Loss","Loss","plot_loss.png")
"""