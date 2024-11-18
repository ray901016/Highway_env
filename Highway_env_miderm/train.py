import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load data
npz_path = "C:/Users/EE715/Desktop/Highway-environment-GradientTape--main/ss/train.npz"
data = np.load(npz_path)
train_data, train_labels = data['data'], to_categorical(data['label'], num_classes=5)

# Initialize distributed training strategy
strategy = tf.distribute.MirroredStrategy()

# Build model
with strategy.scope():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(train_data.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train model
model.fit(train_data, train_labels, epochs=400, batch_size=64, validation_split=0.1, verbose=2)

# Save the model
model.save("optimized_model3.h5")
print("Training completed and model saved.")
