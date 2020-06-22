from keras.models import load_model
import cv2
import numpy as np

filepath = "/code/BaiTapOTruong/AI/image_data/scissors/5mipmm4t8RM5o0tW.png"

CASES  = ["rock", "paper", "scissors", "none"]

#def mapper(val):
#    return REV_CLASS_MAP[val]
def Moves(val):
    return CASES[val]

model = load_model("keo-bua-bao-model.h5")

# prepare the image
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (227, 227))

# predict the move made
pred = model.predict(np.array([img]))
move_code = np.argmax(pred[0])
move_name = Moves(move_code)

print("Predicted: {}".format(pred))
