import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Datasetul shuttle
data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

# Impartim datele 50-50 train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Normalizare min-max
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)
X_train = (X_train - X_min) / (X_max - X_min)
X_test = (X_test - X_min) / (X_max - X_min)

# Autoencoderul
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = keras.Sequential([
            layers.Dense(8, activation='relu'),
            layers.Dense(5, activation='relu'),
            layers.Dense(3, activation='relu')
        ])
        # Decoder
        self.decoder = keras.Sequential([
            layers.Dense(5, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(X_train.shape[1], activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

# Antrenam autoencoderul
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=1024,
    validation_data=(X_test, X_test),
    verbose=1
)

# Reprezentam grafic antrenamentul si validation loss

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Calculam erorile de reconstructie
reconstructed_train = autoencoder(X_train)
reconstructed_test = autoencoder(X_test)

train_errors = np.mean(np.square(X_train - reconstructed_train), axis=1)
test_errors = np.mean(np.square(X_test - reconstructed_test), axis=1)

# Calculam pragul
threshold = np.quantile(train_errors, 0.9)

# Gasim anomaliile
y_train_pred = (train_errors > threshold).astype(int)
y_test_pred = (test_errors > threshold).astype(int)

# Calculam balanced accuracy
train_accuracy = balanced_accuracy_score(y_train, y_train_pred)
test_accuracy = balanced_accuracy_score(y_test, y_test_pred)

print(f"Balanced Accuracy - Train: {train_accuracy:.4f}")
print(f"Balanced Accuracy - Test: {test_accuracy:.4f}")
print(f"Threshold for anomaly detection: {threshold:.4f}")