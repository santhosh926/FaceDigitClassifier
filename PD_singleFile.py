import numpy as np
import time
import matplotlib.pyplot as plt

def read_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    labels = [int(line.strip()) for line in lines]
    num_samples = len(labels)
    return labels, num_samples

def read_images(file_path, num_samples):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    image_width = len(lines[0])
    image_height = len(lines) // num_samples
    images = []
    for idx in range(num_samples):
        image = np.zeros((image_height, image_width))
        for row in range(image_height):
            line = lines[image_height * idx + row]
            for col, char in enumerate(line):
                if char in "+#":
                    image[row, col] = 1
        images.append(image)
    return images

def activate(z):
    return 1 / (1 + np.exp(-z))

def train_model(features, labels, learning_rate=0.5, iterations=50):
    weights = np.random.rand(features.shape[1], 10)
    for iteration in range(iterations):
        error_count = 0
        for idx in range(labels.shape[0]):
            logits = np.dot(features[idx], weights)
            prediction = np.argmax(logits)
            if prediction != labels[idx]:
                weights[:, labels[idx]] += learning_rate * features[idx]
                error_count += 1
        if error_count == 0:
            break
    return weights

def predict(weights, inputs):
    logits = np.dot(inputs, weights)
    predictions = np.argmax(logits, axis=1)
    return predictions

def plot_data(metrics, title, color, ylabel):
    x_axis = np.linspace(0.1, 1.0, 10)
    plt.plot(x_axis, metrics, label='Metric', color=color)
    plt.xlabel('Training Data Fraction')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def prepare_data(image_file, label_file):
    labels, num_samples = read_labels(label_file)
    images = read_images(image_file, num_samples)
    flattened_images = np.array([img.flatten() for img in images])
    labels = np.array(labels)  # Ensure labels are a NumPy array
    indices = np.random.permutation(flattened_images.shape[0])
    return flattened_images[indices], labels[indices]  # Apply indices to both arrays

def read_single_image(image_file, label_file):
    labels, _ = read_labels(label_file)  # Read all labels and pick the first one for our test
    images = read_images(image_file, len(labels))  # Read all images assuming labels align with images
    single_image = images[2].flatten()  # Flatten the first image
    single_label = labels[2]  # Pick the first label
    return single_image, single_label

def calculate_accuracy(predictions, actuals):
    return np.mean(predictions == actuals)

def main():
    training_images_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/digitdata/trainingimages"
    training_labels_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/digitdata/traininglabels"
    testing_images_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/digitdata/testimages"
    testing_labels_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/digitdata/testlabels"

    # Prepare and train on all data
    x_train, y_train = prepare_data(training_images_path, training_labels_path)
    weights = train_model(x_train, y_train)

    # Now test on a single image
    single_image, actual_label = read_single_image(testing_images_path, testing_labels_path)
    single_image = single_image.reshape(1, -1)  # Reshape to fit input requirements of the prediction function
    predicted_label = predict(weights, single_image)
    
    print("Predicted Label:", predicted_label)
    print("Actual Label:", actual_label)

main()

