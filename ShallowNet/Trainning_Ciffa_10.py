# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10 # Vì Keras đã tích hợp dataset dữ liệu cifar10
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Cách dùng Trainning_model_shallownet.py -d datasets/animals

# Bước 1. Chuẩn bị dữ liệu
# Nạp dataset từ đĩa rồi co dãn mức xám của pixel trong vùng [0,1]
print("[INFO] Nạp ảnh...")
# Nạp dữ liệu cifar10 từ keras,và cũng không cần chia tách dữ liệu trainning
# và testing, việc này đã tích hợp trong keras.
((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Chuyển dữ liệu nhãn ở số nguyên vào biểu diễn dưới dạng vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Khởi tạo nhãn cho dữ liệu cifar10 (do dữ liệu cifar 10 có 10 nhãn này)
labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]


# Bước 2. Xây dựng model mạng
# Tạo bộ tối ưu hóa cho mô hỉnh
opt = SGD(learning_rate=0.01)

# Khởi tạo mô hình mạng, biên dịch mô hình
print("[INFO] Tạo mô hình...")
model = ShallowNet.build(width=32, height=32, depth=3, classes=10) #classes =10 vì có 10 nhãn
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# Bước 3. trainning mạng
print("[INFO] training mạng ...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=32, epochs=100, verbose=1)
model.save("model.hdf5")

model.summary()


# Bước 4. Đánh giá mạng
print("[INFO] Đánh giá mạng...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=labelNames)) #labelNames danh sách nhãn đã định nghĩa ở trên

# Vẽ kết quả trainning: sự mất mát (loss) quá trình trainning và độ chính xác (accuracy)
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="Mất mát khi trainning")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="Độ chính xác khi trainning")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Biểu đồ hiển thị mất mát trong Training và độ chính xác")
plt.xlabel("Epoch #")
plt.ylabel("Mất mát/Độ chính xác")
plt.legend()
plt.show()