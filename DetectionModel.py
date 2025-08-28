import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

data_dir = 'Dataset'
categories = ['drown', 'not_drown']

def load_images(data_dir, categories, img_size=(64, 64)):
    images = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                resized_array = cv2.resize(img_array, img_size)
                images.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image: {e}")
                continue
    return np.array(images), np.array(labels)

images, labels = load_images(data_dir, categories)

images = images / 255.0

n_samples, height, width, channels = images.shape
images_flat = images.reshape(n_samples, height * width * channels)

X_train, X_val, y_train, y_val = train_test_split(images_flat, labels, test_size=0.2, random_state= 21)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred, target_names=categories)

print(f'Validation Accuracy: {accuracy * 100:.2f}%')
print(f'Classification Report:\n{report}')

joblib.dump(model, 'drowning_detector.pkl')

'''
def update_accuracy_log(new_accuracy, file_path='test_runs.txt'):
    runs, max_accuracy, min_accuracy, avg_accuracy = 0, 0.0, 100.0, 0.0
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                key, value = line.strip().split(': ')
                if key == 'runs':
                    runs = int(value)
                elif key == 'max_accuracy':
                    max_accuracy = float(value)
                elif key == 'min_accuracy':
                    min_accuracy = float(value)
                elif key == 'avg_accuracy':
                    avg_accuracy = float(value)
    
    runs += 1
    max_accuracy = max(max_accuracy, new_accuracy)
    min_accuracy = min(min_accuracy, new_accuracy)
    avg_accuracy = ((avg_accuracy * (runs - 1)) + new_accuracy) / runs

    with open(file_path, 'w') as file:
        file.write(f"runs: {runs}\n")
        file.write(f"max_accuracy: {max_accuracy}\n")
        file.write(f"min_accuracy: {min_accuracy}\n")
        file.write(f"avg_accuracy: {avg_accuracy}\n")

    print(f"Updated log: runs={runs}, max_accuracy={max_accuracy:.2f}, min_accuracy={min_accuracy:.2f}, avg_accuracy={avg_accuracy:.2f}")

new_accuracy = accuracy * 100
update_accuracy_log(new_accuracy)
'''
