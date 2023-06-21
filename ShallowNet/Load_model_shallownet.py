# import the necessary packages
from preprocessing import imagetoarraypreprocessor
from preprocessing import simplepreprocessor
from datasets import simpledatasetloader

from keras.models import load_model
from imutils import paths
import numpy as np
import cv2
import os
import argparse

#  Load_model_shallownet.py -d <folder chứa ảnh phân loại>
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,help="Nhập vào folder để phân loại")
#args = vars(ap.parse_args())

# Khởi tạo danh sách nhãn
classLabels = ["Apple", "Banana", "Chery","Lemon","Mango","Orange","Pineapple","Rambutan","Strawberry","Tomato"]

sp = simplepreprocessor.SimplePreprocessor(32, 32) # Thiết lập kích thước ảnh 32 x 32
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor() # Gọi hàm để chuyển ảnh sang mảng


# Nạp dataset từ đĩa rồi co dãn mức xám của pixel trong vùng [0,1]
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp, iap])
data, _ = sdl.load(["Image\\lemon.jpg"])
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] Nạp model mạng pre-trained ...")
model = load_model("md1.h5")

# make predictions on the images
print("[INFO] Đang dự đoán để phân lớp...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# Đọc file ảnh
image = cv2.imread("Image\\lemon.jpg")

# Viết label lên ảnh
cv2.putText(image, "label: {}".format(classLabels[preds[0]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Hiển thị ảnh
cv2.imshow("Image", image)
cv2.waitKey(0)







