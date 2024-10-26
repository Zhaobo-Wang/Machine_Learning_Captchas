import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('0991-0.9956.keras')

# Print the model summary to see the architecture
model.summary()