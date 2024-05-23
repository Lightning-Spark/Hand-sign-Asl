import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Load the data
data = []
labels = []
for filename in os.listdir('data'):
    if filename.endswith('.npy'):
        label = filename.split('_')[0]
        labels.append(label)
        data.append(np.load(os.path.join('data', filename)))

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Load the KNN model
filename = 'models/knn_model_k5.joblib'
knn = joblib.load(filename)

# Use the model to predict the hand sign labels
predictions = knn.predict(data)

# Compute the accuracy of the model
accuracy = np.mean(predictions == labels)
print(f'Accuracy: {accuracy:.2f}')

# Evaluate the accuracy of the model on the testing set
accuracy = knn.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

plt.bar(['Accuracy'], [accuracy])
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.title('Hand Sign Detection Model Evaluation')
plt.show()