def classify_signal_1d(filepath, model_path):
    """
    Classify a raw 1D vibration signal using a trained 1D-CNN model.

    This function loads a signal from a given .txt file, applies MinMax normalization,
    reshapes it to match the model's input format, and uses a trained model to predict
    whether the signal corresponds to a healthy or faulty device.

    Args:
        filepath (str): Path to the input .txt file containing the raw signal.
        model_path (str): Path to the saved 1D-CNN model (in .h5 format).

    Returns:
        tuple: A tuple (predicted_class, confidence) where:
            - predicted_class (int): 0 for healthy, 1 for faulty.
            - confidence (float): Confidence score (between 0 and 1) of the prediction.

    Example:
        >>> classify_signal_1d("data/sample_signal.txt", "models/best_1D_model.h5")
        🧠 Prediction: Healthy (94.28%)
        (0, 0.9428)
    """
    from preprocessing import load_signal, normalize_signal, fix_signal_length
    import tensorflow as tf
    import numpy as np

    model = tf.keras.models.load_model(model_path)
    signal = normalize_signal(fix_signal_length(load_signal(filepath)))
    signal = np.expand_dims(signal[:25000], axis=(0, -1))  # (1, 25000, 1)

    prediction = model.predict(signal)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    label_map = {0: "Healthy", 1: "Faulty"}
    print(f"🧠 Prediction: {label_map[predicted_class]} ({confidence:.2%})")
    return predicted_class, confidence


def classify_signal_2d(filepath, model_path):
    """
    Classify a 1D vibration signal as 'Healthy' or 'Faulty' using a trained 2D CNN model.

    This function performs the following steps:
    1. Loads a raw signal from a .txt file (ignoring metadata lines starting with '%').
    2. Fixes the signal length to a standard size (default 25000).
    3. Converts the fixed signal into a spectrogram (2D representation).
    4. Reshapes the spectrogram to match the input format of the trained 2D CNN model.
    5. Predicts the class using the model and prints the predicted label and confidence.

    Args:
        filepath (str): Path to the input .txt file containing the signal.
        model_path (str): Path to the saved Keras model (HDF5 format).

    Returns:
        Tuple[int, float]: Predicted class (0 = Healthy, 1 = Faulty), and confidence score.
    """
    from preprocessing import load_signal_and_fs, normalize_signal, signal_to_spectrogram, fix_signal_length
    import tensorflow as tf
    import numpy as np

    model = tf.keras.models.load_model(model_path)
    signal, fs = load_signal_and_fs(filepath)
    signal = fix_signal_length(signal)
    # signal = normalize_signal(load_signal(filepath))
    spectrogram = signal_to_spectrogram(signal, fs)

    input_data = np.expand_dims(spectrogram, axis=(0, -1))
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    label_map = {0: "Healthy", 1: "Faulty"}
    print(f"🧠 Prediction: {label_map[predicted_class]} ({confidence:.2%})")
    return predicted_class, confidence
