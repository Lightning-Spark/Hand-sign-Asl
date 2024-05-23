import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the KNN model
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Evaluate the accuracy of the model on the testing set
accuracy = []
accuracy.append(knn.score(X_test, y_test))
print(f'Accuracy: {accuracy[0]:.2f}')

# plt.bar(['Accuracy'], [accuracy[0]])
# plt.ylim([0, 1])
# plt.ylabel('Accuracy')
# plt.title('Hand Sign Detection Model Evaluation')
# plt.show()

# Save the model
if not os.path.exists('models'):
    os.makedirs('models')
filename = f'models/knn_model_k{k}.joblib'
joblib.dump(knn, filename)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Train the KNN model
knn.fit(X_train, y_train)

# Evaluate the accuracy of the model on the testing set

accuracy.append(knn.score(X_test, y_test))
print(f'Accuracy: {accuracy[1]:.2f}')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)

# Train the KNN model
knn.fit(X_train, y_train)

# Evaluate the accuracy of the model on the testing set
accuracy.append(knn.score(X_test, y_test))
print(f'Accuracy: {accuracy[2]:.2f}')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=42)

# Train the KNN model
knn.fit(X_train, y_train)

# Evaluate the accuracy of the model on the testing set
accuracy.append(knn.score(X_test, y_test))
print(f'Accuracy: {accuracy[3]:.2f}')
x = [.2,.3,.4,.5]

# Create a figure and axis object
fig, ax = plt.subplots()

# Plotting the bar chart
ax.bar(x, accuracy)

# Setting the chart title and axes labels
ax.set_title('Accuracy Scores for KNN')
ax.set_xlabel('x')
ax.set_ylabel('Accuracy')


#Plotting 
# plt.plot([x], [accuracy],color='green')
# plt.ylim([0,1])
# plt.ylabel('Accuracy')
# plt.title('Hand Sign Detection Model Evaluation')
plt.show()