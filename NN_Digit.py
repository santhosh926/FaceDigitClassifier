import numpy as np
import time
import matplotlib.pyplot as plt

def read_labels(file_path):
    with open(file_path) as file:
        lines = file.readlines()
    labels = [int(line.strip()) for line in lines]
    num_samples = len(labels)
    return labels, num_samples

def read_samples(file_path, num_samples):
    with open(file_path) as file:
        lines = file.readlines()
    image_width = len(lines[0])
    image_height = int(len(lines) / num_samples)
    images = []
    for i in range(num_samples):
        image = np.zeros((image_height, image_width))
        for idx, line in enumerate(lines[i * image_height:(i + 1) * image_height]):
            for j, char in enumerate(line):
                if char in "+#":
                    image[idx, j] = 1
        images.append(image)
    return images

def convert_one_hot(labels):
    one_hot_labels = []
    for label in labels:
        encoded = np.zeros(10)
        encoded[label] = 1
        one_hot_labels.append(encoded)
    return np.array(one_hot_labels)

def prepare_data(image_file, label_file):
    labels, num_samples = read_labels(label_file)
    images = read_samples(image_file, num_samples)
    labels = convert_one_hot(labels)
    flattened_images = [image.flatten() for image in images]
    indices = np.random.permutation(len(flattened_images))
    return np.array(flattened_images)[indices], labels[indices]

def update_weights(weights, biases, inputs, targets, iterations, learning_rate):
    for _ in range(iterations):
        gradient_w, gradient_b, cost = compute_gradients(weights, biases, inputs, targets)
        weights -= learning_rate * gradient_w
        biases -= learning_rate * gradient_b
    return weights, biases

def compute_gradients(weights, biases, inputs, targets):
    num_examples = inputs.shape[0]
    predictions = sigmoid(np.dot(inputs, weights) + biases).reshape(-1, 10)
    cost = -np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)) / num_examples
    gradient_w = np.dot(inputs.T, (predictions - targets)) / num_examples
    gradient_b = np.sum(predictions - targets, axis=0) / num_examples
    return gradient_w, gradient_b, cost

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def make_predictions(weights, biases, inputs):
    logits = np.dot(inputs, weights) + biases
    probabilities = sigmoid(logits)
    return (probabilities > 0.5).astype(int)

def calculate_accuracy(predictions, labels):
    return np.mean(np.all(predictions == labels, axis=1))

def train_model(train_inputs, train_labels, iterations=2000, learning_rate=0.5):
    num_features = train_inputs.shape[1]
    weights = np.zeros((num_features, 10))
    biases = np.zeros(10)
    weights, biases = update_weights(weights, biases, train_inputs, train_labels, iterations, learning_rate)
    return weights, biases

def plot_results(values, title, color, label):
    x_axis = np.arange(0.1, 1.1, 0.1)
    plt.plot(x_axis, values, label=label, color=color)
    plt.xlabel('Fraction of Training Data')
    plt.title(title)
    plt.ylabel(label)
    plt.tight_layout()
    plt.show()

def main():
    train_images_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/digitdata/trainingimages"
    train_labels_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/digitdata/traininglabels"
    test_images_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/digitdata/testimages"
    test_labels_path = "/Users/santhosh/Desktop/Intro to AI/Final Project/data/digitdata/testlabels"
    
    train_inputs, train_labels = prepare_data(train_images_path, train_labels_path)
    test_inputs, test_labels = prepare_data(test_images_path, test_labels_path)
    
    training_times = []
    accuracies = []
    for i in range(1, 11):
        start_time = time.time()
        weights, biases = train_model(train_inputs[:int(i * len(train_inputs) / 10)], train_labels[:int(i * len(train_labels) / 10)])
        elapsed_time = time.time() - start_time
        predictions = make_predictions(weights, biases, test_inputs)
        accuracy = calculate_accuracy(predictions, test_labels)
        
        training_times.append(elapsed_time)
        accuracies.append(accuracy)
        print(f"Test accuracy after training with {i*10}% of data: {accuracy:.2%}")

    plot_results(training_times, 'Training Time', 'blue', 'Time (seconds)')
    plot_results(accuracies, 'Model Accuracy', 'red', 'Accuracy')

main()
