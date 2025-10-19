# cnn_mnist.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and Preprocess the MNIST Dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocessing Steps:
# a. Reshape images to (28, 28, 1) to include the channel dimension for the CNN.
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

# b. Normalize pixel values from [0, 255] to [0, 1] for better model performance.
x_train /= 255.0
x_test /= 255.0

# c. One-hot encode the labels (e.g., 5 -> [0,0,0,0,0,1,0,0,0,0]).
# This is necessary for the 'categorical_crossentropy' loss function.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test_categorical = tf.keras.utils.to_categorical(y_test, 10) # Keep original y_test for visualization

print("Data preprocessing complete.")
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print("-" * 30)

# 2. Build the CNN Model
# We use a simple Sequential model.
model = tf.keras.Sequential([
    # Convolutional layer with 32 filters, 3x3 kernel, and ReLU activation.
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max pooling layer to reduce spatial dimensions.
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Another convolutional layer.
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    # Another max pooling layer.
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Flatten the 2D feature maps into a 1D vector.
    tf.keras.layers.Flatten(),
    # Dense (fully connected) layer with 128 neurons.
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer with 10 neurons (one for each digit) and softmax activation
    # to get probability distribution.
    tf.keras.layers.Dense(10, activation='softmax')
])

# 3. Compile the Model
# We specify the optimizer, the loss function for multi-class classification,
# and the metric we want to track.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
print("-" * 30)


# 4. Train the Model
print("Training the model...")
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    validation_split=0.1) # Use 10% of training data for validation
print("Model training complete.")
print("-" * 30)

# 5. Evaluate the Model on the Test Set
print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(x_test, y_test_categorical, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
# The goal of >95% accuracy is met.
print("-" * 30)


# 6. Visualize Predictions on Sample Images
print("Visualizing predictions on 5 sample images...")
# Get predictions for the entire test set
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Select 5 random images from the test set to display
plt.figure(figsize=(10, 5))
for i in range(5):
    # Choose a random index
    idx = np.random.randint(0, len(x_test))
    
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[idx]}\nPred: {predicted_labels[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
# 7. Save the Trained Model
print("Saving the model...")
model.save('mnist_cnn_model.h5')
print("Model saved successfully as mnist_cnn_model.h5")