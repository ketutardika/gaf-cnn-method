import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix

# 1. Data Processing Module
def improved_gaf(data, size=32):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    phi = np.arccos(data)
    r, c = np.meshgrid(range(len(data)), range(len(data)))
    gaf = np.cos(phi[r] + phi[c])
    return np.resize(gaf, (size, size)).astype(np.float32)

# 2. Data Augmentation Module
def improved_ddpm(image, noise_steps=5):
    for _ in range(noise_steps):
        noise = np.random.normal(0, 0.1, image.shape).astype(np.float32)
        image = image + noise
    return np.clip(image, 0, 1)

# 3. Image Classification Module (ETNet v2 simplified as CNN)
def create_etnet_v2(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),  # Regularization layer

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),  # Regularization layer

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Flatten(),
        Dropout(0.5),  # Regularization layer

        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Data Generator
class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        X_batch = self.X[indexes]
        y_batch = self.y[indexes]

        X_batch_img = np.array([improved_gaf(x) for x in X_batch])
        X_batch_aug = np.array([improved_ddpm(x) for x in X_batch_img])
        return X_batch_aug.reshape((-1, 32, 32, 1)), y_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Main process
def main():
    # Load data (first 50,000 rows)
    data = pd.read_csv('02-15-2018.csv', nrows=50000)
    
    # Select features and target
    features = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts']
    X = data[features]
    y = data['Label']  # Assuming 'Label' column exists
    
    # Handle non-numeric data
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors='coerce')
    X = X.fillna(X.mean())
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create data generators
    train_generator = DataGenerator(X_train_scaled, y_train, batch_size=16)
    test_generator = DataGenerator(X_test_scaled, y_test, batch_size=16, shuffle=False)
    
    # Create and compile model
    model = create_etnet_v2((32, 32, 1), num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    history = model.fit(train_generator, epochs=10, validation_data=test_generator, verbose=1)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # Predict using the test generator
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print(classification_report(y_test, y_pred_classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
