import cv2
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_resized = np.array([cv2.resize(img, (28, 28)) for img in x_train])
x_test_resized = np.array([cv2.resize(img, (28, 28)) for img in x_test])

x_train_resized = x_train_resized.astype('float32') / 255.0
x_test_resized = x_test_resized.astype('float32') / 255.0

x_train_resized = np.expand_dims(x_train_resized, axis=-1)
x_test_resized = np.expand_dims(x_test_resized, axis=-1)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Prevents overfitting
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer (10 digits)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_resized, y_train, epochs=5, validation_data=(x_test_resized, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test_resized, y_test)
print("Test Accuracy:", accuracy)
print("Test Loss:", loss)

# Save the trained model
model.save('digits.keras')
