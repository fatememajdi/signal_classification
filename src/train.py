import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import ParameterSampler
layers = tf.keras.layers
models = tf.keras.models


def create_model(input_length=25000, num_filters=64, dropout_rate=0.3, max_pool=5):
    """
    Build and return a 1D Convolutional Neural Network for binary classification.

    The model consists of 3 Conv1D layers with increasing filters and decreasing kernel sizes,
    followed by MaxPooling, Dropout, and GlobalAveragePooling for dimensionality reduction.
    Final classification is done with a Dense layer and softmax activation.

    Args:
        input_length (int): Length of the input 1D signal.
        num_filters (int): Number of filters for the first convolutional layer.
                           The next conv layers use 2x and 4x this value.
        dropout_rate (float): Dropout rate used after the second convolution.
        max_pool (int): Kernel size for the MaxPooling1D layer.

    Returns:
        tf.keras.Model: Compiled Sequential Keras model ready for training.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_length, 1)),
        tf.keras.layers.Conv1D(num_filters, 9, strides=2, activation='relu'),
        tf.keras.layers.MaxPool1D(max_pool, strides=2),
        tf.keras.layers.Conv1D(num_filters*2, 7, strides=2, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv1D(num_filters*4, 5, strides=2, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model


def create_model_2d(input_shape, num_filters=32, dropout_rate=0.5):
    """
    Build and return a 2D Convolutional Neural Network for binary classification.

    This model is designed for classifying 2D representations of signals
    (e.g., spectrograms or STFTs). It includes two Conv2D layers with ReLU activation,
    followed by Batch Normalization, MaxPooling, Dropout for regularization,
    and two fully connected layers for final classification.

    Args:
        input_shape (tuple): Shape of the input data (height, width, channels),
                             e.g., (128, 128, 1) for grayscale spectrograms.
        num_filters (int): Number of filters in the convolutional layers.
                           Both Conv2D layers use this value.
        dropout_rate (float): Dropout rate applied before the final Dense layers
                              to reduce overfitting.

    Returns:
        tf.keras.Model: A compiled Keras Sequential model ready for training.

    Example:
        >>> model = create_model_2d((128, 128, 1), num_filters=32, dropout_rate=0.5)
        >>> model.summary()
    """
    model = models.Sequential([
        layers.Conv2D(num_filters, (3, 3), activation='relu',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(num_filters, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    return model


def random_search_hyperparameters_2d(X_train, y_train, X_test, y_test):
    """
    Perform random search to tune hyperparameters for a 2D CNN model.

    This function performs randomized hyperparameter search over a predefined grid
    to find the best-performing 2D CNN configuration on the given training and test data.
    For each sampled configuration, the model is trained and evaluated,
    and the best model is saved based on validation accuracy.

    Args:
        X_train (np.ndarray): Training input data of shape (num_samples, height, width, channels).
        y_train (np.ndarray): Corresponding labels for training data.
        X_test (np.ndarray): Test input data.
        y_test (np.ndarray): Test labels.

    Saves:
        best_2DCNN_model_randomsearch.h5: The best-performing trained model based on validation accuracy
                                          is saved to the ../models/ directory.

    Prints:
        - Training and validation curves for each configuration
        - Best hyperparameters and final validation accuracy

    Example:
        >>> random_search_hyperparameters_2d(X_train, y_train, X_test, y_test)
    """
    param_grid = {
        'lr': [1e-4, 1e-5],
        'batch_size': [16, 32],
        'num_filters': [32, 16],
        'epochs': [50, 60]
    }

    random_grid = list(ParameterSampler(param_grid, n_iter=5, random_state=42))

    best_accuracy = 0
    best_model = None
    best_params = None

    for i, params in enumerate(random_grid):
        print(f"\n🔍 Running config {i+1}/5: {params}")

        model = create_model_2d(input_shape=X_train.shape[1:],
                                num_filters=params['num_filters'])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=params['epochs'],
                            batch_size=params['batch_size'],
                            verbose=0,
                            shuffle=True)

        val_acc = history.history['val_accuracy'][-1]
        print(f"✅ Validation Accuracy: {val_acc:.2%}")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title(f"Accuracy - Config {i+1}")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f"Loss - Config {i+1}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model
            best_params = params

    best_model.save("../models/best_2DCNN_model_randomsearch.h5")
    print(f"\n🏆 Best parameters: {best_params}")
    print(f"✅ Best validation accuracy: {best_accuracy:.2%}")


def random_search_hyperparameters(X_train, y_train, X_test, y_test):
    """
    Perform random search over a set of hyperparameters for a 1D CNN model.

    This function searches different combinations of learning rate, batch size,
    number of filters, and epochs. It trains a model for each configuration, tracks
    validation accuracy, and saves the best-performing model to disk.

    Args:
        X_train (np.ndarray): Training input signals of shape (num_samples, signal_length).
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test input signals.
        y_test (np.ndarray): Test labels.

    Returns:
        None. Saves the best model to 'best_1D_model_randomsearch.h5' and prints best params.
    """
    param_grid = {
        'lr': [1e-4, 1e-5],
        'batch_size': [32],
        'num_filters': [32],
        'epochs': [80, 90],
        'max_pool': [5]
    }
    random_grid = list(ParameterSampler(param_grid, n_iter=5, random_state=42))

    best_accuracy = 0
    best_model = None
    best_params = None

    for i, params in enumerate(random_grid):
        print(f"\n[🔍] Running random config {i+1}/5: {params}")

        model = create_model(
            input_length=X_train.shape[1], num_filters=params['num_filters'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=params['epochs'],
                            batch_size=params['batch_size'],
                            verbose=0,
                            shuffle=True)

        val_acc = history.history['val_accuracy'][-1]
        print(f"Validation Accuracy: {val_acc:.2%}")
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title(f"Accuracy - Config {i+1}")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f"Loss - Config {i+1}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model
            best_params = params

    best_model.save("../models/best_1D_model_randomsearch.h5")
    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print("Model saved to: best_1D_model_randomsearch.h5")
