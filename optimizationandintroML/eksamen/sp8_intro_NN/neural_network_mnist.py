# Import TensorFlow and print the version being used
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Load the MNIST dataset - contains 70,000 grayscale images of handwritten digits (60k train, 10k test)
mnist = tf.keras.datasets.mnist

# Split the dataset into training and testing sets, and normalize pixel values to 0-1 range
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalization helps with convergence

# Build the neural network model using Sequential API
model = tf.keras.models.Sequential([
    # Flatten 28x28 images to 784-dimensional vector input
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # First hidden layer with 128 neurons and ReLU activation function
    # ReLU helps with non-linear transformations and prevents vanishing gradient
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout layer randomly sets 20% of neurons to 0 during training to prevent overfitting
    tf.keras.layers.Dropout(0.2),
    
    # Output layer with 10 neurons (one for each digit 0-9) using linear activation (logits)
    tf.keras.layers.Dense(10)
])

# Make initial prediction on first training sample (before training) to demonstrate raw outputs
predictions = model(x_train[:1]).numpy()
predictions  # These are "logits" - unnormalized predictions

# Convert logits to probabilities using Softmax (values sum to 1)
tf.nn.softmax(predictions).numpy()

# Define loss function - SparseCategoricalCrossentropy (for integer labels)
# from_logits=True tells TF the model outputs raw logits instead of probabilities
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Calculate initial loss (before training) as baseline
loss_fn(y_train[:1], predictions).numpy()

# Compile the model with Adam optimizer, loss function, and track accuracy metric
# Adam: Adaptive Moment Estimation (combines RMSProp and Momentum)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train the model for 5 epochs (full passes through the training data)
# Each epoch shows loss and accuracy metrics
model.fit(x_train, y_train, epochs=5)

# Evaluate model performance on test set (unseen data)
# verbose=2 suppresses progress bar but shows final metrics
model.evaluate(x_test,  y_test, verbose=2)

# Create a new model that adds Softmax layer to convert logits to probabilities
# This is useful for interpretation and final predictions
probability_model = tf.keras.Sequential([
    model,  # Original model (including weights)
    tf.keras.layers.Softmax()  # Add probability conversion
])

# Get probability distributions for first 5 test samples
probability_model(x_test[:5])