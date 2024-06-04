import numpy as np
import time
import matplotlib.pyplot as plt

def fetch_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    labels = [int(line.strip()) if int(line.strip()) > 0 else -1 for line in lines]
    return labels, len(labels)

def fetch_samples(file_path, num_samples, pool_size):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    image_width = len(lines[0])
    image_height = len(lines) // num_samples
    images = []
    for i in range(num_samples):
        image = np.zeros((image_height, image_width))
        for row in range(image_height):
            line = lines[i * image_height + row]
            image[row] = [1 if char in "+#" else 0 for char in line]
        images.append(image)
    
    # Pooling
    pooled_images = pool_images(images, image_height, image_width, pool_size)
    return np.array(pooled_images)

def pool_images(images, image_height, image_width, pool_size):
    pooled_images = []
    new_height, new_width = image_height // pool_size, image_width // pool_size
    for image in images:
        pooled_image = np.zeros((new_height, new_width))
        for i in range(new_height):
            for j in range(new_width):
                pool_region = image[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                pooled_image[i, j] = np.sum(pool_region)
        pooled_images.append(pooled_image)
    return pooled_images

def train_perceptron(features, labels, learning_rate=0.5, max_iterations=2000):
    weights = np.random.rand(features.shape[1])
    bias = 0
    for _ in range(max_iterations):
        errors = 0
        for feature, label in zip(features, labels):
            activation = np.dot(feature, weights) + bias
            if label * activation <= 0:
                weights += learning_rate * feature * label
                bias += learning_rate * label
                errors += 1
        if errors == 0:
            break
    return weights, bias

def predict(weights, bias, features):
    predictions = np.dot(features, weights) + bias
    return np.where(predictions > 0, 1, -1)

def calculate_accuracy(predictions, actuals):
    return np.mean(predictions == actuals)

def plot_results(values, title, color, ylabel):
    x_axis = np.linspace(0.1, 1.0, 10)
    plt.plot(x_axis, values, label='Metric', color=color)
    plt.xlabel('Fraction of Training Data')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def preprocess_data(image_file, label_file, pool_size):
    labels, num_samples = fetch_labels(label_file)
    images = fetch_samples(image_file, num_samples, pool_size)
    flattened_images = np.array([img.flatten() for img in images])
    idx = np.random.permutation(flattened_images.shape[0])
    return flattened_images[idx], np.array(labels)[idx]

def read_single_image_and_label(image_file, label_file, pool_size):
    labels, num_samples = fetch_labels(label_file)  # Read labels and assume there is at least one label
    images = fetch_samples(image_file, num_samples, pool_size)  # Fetch and pool images
    single_image = images[6].flatten()  # Flatten the first image
    single_label = labels[6]  # The first label
    return single_image, single_label

def main():
    pool_size = 2
    training_images_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/facedata/facedatatrain"
    training_labels_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/facedata/facedatatrainlabels" 
    testing_images_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/facedata/facedatatest"
    testing_labels_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/facedata/facedatatestlabels"

    x_train, y_train = preprocess_data(training_images_path, training_labels_path, pool_size)
    weights, bias = train_perceptron(x_train, y_train)

    # Now test on a single image
    single_image, actual_label = read_single_image_and_label(testing_images_path, testing_labels_path, pool_size)
    single_image = single_image.reshape(1, -1)  # Reshape to fit input requirements of the prediction function
    predicted_label = predict(weights, bias, single_image)
    
    print("Predicted Label:", predicted_label)
    print("Actual Label:", actual_label)

if __name__ == "__main__":
    main()
