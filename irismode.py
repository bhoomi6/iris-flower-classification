import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = sns.load_dataset('iris')

# Encode labels
mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
df['species'] = df['species'].map(mapping)

# Features and labels
X = df.drop('species', axis=1)
y = tf.keras.utils.to_categorical(df['species'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=10)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Create images folder
import os
if not os.path.exists('images'):
    os.makedirs('images')

# Save graphs
plt.plot(history.history['accuracy'])
plt.title('Accuracy')
plt.savefig('images/accuracy.png')
plt.clf()

plt.plot(history.history['loss'])
plt.title('Loss')
plt.savefig('images/loss.png')
plt.clf()

# Prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)
print("Predicted class:", np.argmax(prediction))