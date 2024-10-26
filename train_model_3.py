import tensorflow as tf
from model_definition import create_resnet_model
from data_preprocessing import train_dataset, val_dataset

model = create_resnet_model((224, 224, 3))

# Train the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Evaluate the model
model.evaluate(val_dataset)

# Save the model
model.save('captcha_resnet_model.h5')