# Đây là code để sinh ra ảnh tăng cường
# import các thư viện
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import img_to_array
from tensorflow.keras.utils import load_img
import numpy as np
from imutils import paths
import argparse

# Xây dựng các tham số thực thi dòng lệnh
# Use: python augmentation_demo.py -i <fiel ảnh> -o <folder lưu ảnh> -p <tiền tố file ảnh tăng cường>

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Đường dẫn input image")
ap.add_argument("-o", "--output", required=True, help="Đường dẫn output directory để lưu trữ ảnh cần tăng cường")
ap.add_argument("-p", "--prefix", type=str, default="image", help="File tiền tố output đã tăng cường")
args = vars(ap.parse_args())
# nạp ảnh đầu vào, convert it sang mảng NumPy array, rồi reshape nó
print("[INFO] Nạp image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Tạo bộ sinh ảnh tăng cường và khởi tạo tổng số ảnh được sinh ra
aug = ImageDataGenerator(
                          rescale=1./255,
                          rotation_range=40,
                          width_shift_range=0.2,
                          height_shift_range=0.2,
                          shear_range=0.2,
                          zoom_range=0.2,
                          horizontal_flip=True,
                          fill_mode='nearest')
total =

# Sinh ảnh tăng cường
print("[INFO] generating images...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
save_prefix=args["prefix"], save_format="jpg")

# Lặp qua các ảnh đã được tăng cường ảnh trong imageGen
for image in imageGen:
    # Tăng bộ đếm
    total += 1
    # Lặp 10 lần
    # 10 là Số ảnh được sinh ra, thay đổi giá trị này để có được số ảnh mong muốn
    if total == 10:
        break


