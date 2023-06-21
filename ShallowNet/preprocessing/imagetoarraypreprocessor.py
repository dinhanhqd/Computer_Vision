 # import các gói thư viện cần thiết
#from keras_preprocessing.image import img_to_array     --> Phiên bản Keras cũ
from keras.utils.image_utils import img_to_array        # --> Phiên bản năm 2022

class ImageToArrayPreprocessor:  # Tạo lớp để chuyển ảnh --> mảng
    def __init__(self, dataFormat=None):
        # Lưu ảnh đã được định dạng
        self.dataFormat = dataFormat

    def preprocess(self, image): # Định nghĩa phương thức preprocess trả về mảng
        # Hàm img_to_array của Keras
        return img_to_array(image, data_format=self.dataFormat)
