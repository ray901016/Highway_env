import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load training data
train_npz_path = r"C:\Users\EE715\Documents\Highway_env_miderm\train.npz"
train_data = np.load(train_npz_path)
train_features, train_labels = train_data['data'], to_categorical(train_data['label'], num_classes=5)

# Load validation data
val_npz_path = r"C:\Users\EE715\Documents\Highway_env_miderm\validation.npz"
val_data = np.load(val_npz_path)
val_features, val_labels = val_data['data'], to_categorical(val_data['label'], num_classes=5)

# Initialize distributed training strategy
strategy = tf.distribute.MirroredStrategy()

# Build model
with strategy.scope():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(train_features.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train model with validation data
model.fit(
    train_features, train_labels,
    epochs=200,
    batch_size=64,
    validation_data=(val_features, val_labels),
    verbose=2
)

# Save the model
model.save("optimized_model.h5")
print("Training completed and model saved.")