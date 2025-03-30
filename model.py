import cv2
import numpy as np
import tensorflow as tf

# Load and normalize MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.13),
    tf.keras.layers.Dense(10)
])

# Compile the model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test, verbose=2)

# Wrap with softmax for prediction
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# Make predictions on first 5 test images
predictions = probability_model.predict(x_test[:5])
print(predictions)
model.save("digits_model.h5")