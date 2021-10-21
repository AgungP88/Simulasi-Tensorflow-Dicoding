# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf


def solution_C2():
    mnist = tf.keras.datasets.mnist

    # YOUR CODE HERE
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64)
    Categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images = train_images/255.0
    test_images = test_images/255.0
    
    model = tf.keras.models.Sequential([
                          tf.keras.layers.Flatten(input_shape=(28, 28)),
                          tf.keras.layers.Dense(128, activation=tf.nn.relu),
                          tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                          ])
    
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(train_images, 
              train_labels, 
              validation_data = (test_images, test_labels),
              verbose = 1,
              epochs=10)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    if __name__ == '__main__':
        model = solution_C2()
        model.save("model_C2.h5")