import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
#chọn thuật toán train
from keras.optimizers import Adam
#chứa các function cần thiết giúp ta xử lý data nhanh hơn.
from keras.utils import np_utils
#các lớp layer dùng để tạo model
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
#khởi tạo model bằng  Sequential
from keras.models import Sequential
#thao tác với các thư mục
import os

IMG_SAVE_PATH = "image_data"

CASES = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}

NUM_CASES = len(CASES)

def Moves(val):
    return CASES[val]

#khởi tạo model
def getModel():
    model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),
        #xác suất của một nơ-ron sẽ bị loại bỏ trong quá trình training tránh quá tải
        Dropout(0.5),
        #lấy đặc tính từ ảnh
        Convolution2D(NUM_CASES, (1, 1), padding='valid'),
        #relu là hàm kích hoạt, giúp giảm chi phí tính toán
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model

# load ảnh từ dataset
dataset = []
#liệt kê các tệp trong đường dẫn
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    #liệt kê các tệp trong đường dẫn
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        # để không cho file ẩn cản trở 
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])


#tạo một vòng lặp dạng danh sách
data, labels = zip(*dataset)
#chuyển đổi thành dạng danh sách
labels = list(map(Moves, labels))

# one hot encode the labels
#covert class sang binary class matrix, xử lý data nhanh hơn
labels = np_utils.to_categorical(labels)

# định nghĩa model
model = getModel()
model.compile(
    #thuật toán train Adam với tỷ lệ học 0,0001
    optimizer=Adam(lr=0.0001),
    loss ='categorical_crossentropy',
    #độ chính xác
    metrics=['accuracy']
)

# start training với 10 chu kỳ
model.fit(np.array(data), np.array(labels), epochs=10)

# lưu model sau khi train
model.save("keo-bua-bao-model.h5")
