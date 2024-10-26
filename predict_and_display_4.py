import cv2
import numpy as np
from keras.models import load_model
from multicolorcaptcha import CaptchaGenerator

# Load model
model_name = 'captcha_resnet_model.h5'
model_input_shape = (224, 224, 3)
print(f'Load model: {model_name}')
model = load_model(model_name)
model.summary()

# Initialize CaptchaGenerator
captcha_generator = CaptchaGenerator(0)

while True:
    captcha = captcha_generator.gen_captcha_image(difficult_level=0)
    img = cv2.resize(np.asarray(captcha.image), model_input_shape[0:2])[:,:,:3]
    target = captcha.characters

    y = model.predict(img.reshape(1, *model_input_shape))
    labels = [np.argmax(y[i][0]) for i in range(4)]

    img = img.astype(np.uint8).copy()
    info = f'Read: {"".join([str(i) for i in labels])}. Target: {target}.'
    cv2.putText(img, info, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow('Captcha Recognizer', img)
    print(f'Result = {labels}')

    key = cv2.waitKey()
    if key == ord('q'):
        break
    elif key == ord('n'):
        continue