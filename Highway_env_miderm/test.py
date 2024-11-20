import tensorflow as tf
import numpy as np

# Load validation data
validation_data = np.load('validation.npz')
validation_features = validation_data['data']
validation_labels = validation_data['label']

# One-hot encode the labels
num_classes = 5  # Adjust based on your dataset
validation_labels_one_hot = tf.keras.utils.to_categorical(validation_labels, num_classes=num_classes)

# Load the saved model
model = tf.keras.models.load_model('optimized_model.h5')
print("Model loaded from 'optimized_model.h5'")

# Create validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels_one_hot))
validation_dataset = validation_dataset.batch(64)

# Compute accuracy
loss, accuracy = model.evaluate(validation_dataset)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
