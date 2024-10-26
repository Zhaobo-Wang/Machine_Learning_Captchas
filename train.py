import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np
import os
import cv2

num_symbols = 36  
img_shape = (50, 200, 1)  
symbols = "0123456789abcdefghijklmnopqrstuvwxyz"

def create_model():
    img = layers.Input(shape=img_shape) 

    conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D((2, 2), padding='same')(conv1)
    
    conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D((2, 2), padding='same')(conv2)
    
    conv3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(mp2)
    mp3 = layers.MaxPooling2D((2, 2), padding='same')(conv3)
    
    bn = layers.BatchNormalization()(mp3)
    conv4 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(bn)
    mp4 = layers.MaxPooling2D((2, 2), padding='same')(conv4)
    
    # 扁平化并创建5个分支，每个分支预测一个字母
    flat = layers.Flatten()(mp4)
    outs = []
    for _ in range(5):
        dens1 = layers.Dense(128, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1)
        res = layers.Dense(num_symbols, activation='sigmoid')(drop)
        outs.append(res)
    
    # 编译模型并返回
    model = Model(img, outs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

# 数据预处理函数
def preprocess_data(data_dir):
    n_samples = len(os.listdir(data_dir))
    X = np.zeros((n_samples, 50, 200, 1))  # 输入图像数据
    y = np.zeros((5, n_samples, num_symbols))  # 标签数据

    for i, pic in enumerate(os.listdir(data_dir)):
        img = cv2.imread(os.path.join(data_dir, pic), cv2.IMREAD_GRAYSCALE)
        pic_target = pic[:-4]
        if len(pic_target) < 6:
            if img is not None:
                # 打印调试信息
                print(f"Original shape of image {pic}: {img.shape}")
                
                img = cv2.resize(img, (200, 50))  # 调整图像大小为 (200, 50)
                print(f"Resized shape of image {pic}: {img.shape}")
                
                img = img / 255.0
                img = np.reshape(img, (50, 200, 1))
                
                targs = np.zeros((5, num_symbols))
                for j, l in enumerate(pic_target):
                    ind = symbols.find(l)
                    targs[j, ind] = 1
                X[i] = img
                y[:, i] = targs
    
    return X, y


train_dir = 'capts_train'
X_train, y_train = preprocess_data(train_dir)


val_dir = 'capts_val'
X_val, y_val = preprocess_data(val_dir)


model = create_model()


model.summary()


history = model.fit(X_train, [y_train[i] for i in range(5)], epochs=10, batch_size=32, validation_data=(X_val, [y_val[i] for i in range(5)]))

model.save('my_model.keras')

print(f"Training accuracy: {history.history['accuracy'][-1]}")
print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")