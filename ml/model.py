"""TensorFlow model factories; importing this module does not import TensorFlow."""

from __future__ import annotations

from typing import Sequence

from .preprocess import IMAGE_SIZE


def _keras():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for model construction and training. "
            "Install backend/requirements.txt."
        ) from exc
    return tf.keras


_LOSS_CACHE: dict[str, object] = {}


def smoothed_sparse_crossentropy_class():
    """Return a registered, serializable label-smoothed sparse CE loss class.

    Keras 3's ``SparseCategoricalCrossentropy`` has no ``label_smoothing`` option,
    and a plain Python closure cannot be deserialized from a ``.keras`` file.
    Registering a proper ``Loss`` subclass keeps the trained model loadable via
    :func:`load_keras_model` (which re-registers it on the loading side).
    """
    if "loss" in _LOSS_CACHE:
        return _LOSS_CACHE["loss"]
    import tensorflow as tf

    keras = _keras()

    class SmoothedSparseCategoricalCrossentropy(keras.losses.Loss):
        def __init__(self, label_smoothing: float = 0.0, **kwargs):
            super().__init__(**kwargs)
            self.label_smoothing = float(label_smoothing)

        def call(self, y_true, y_pred):
            num_classes = y_pred.shape[-1]
            targets = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
            targets = targets * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
            return keras.losses.categorical_crossentropy(targets, y_pred)

        def get_config(self):
            config = super().get_config()
            config["label_smoothing"] = self.label_smoothing
            return config

    keras.utils.register_keras_serializable(package="signtrans")(
        SmoothedSparseCategoricalCrossentropy
    )
    _LOSS_CACHE["loss"] = SmoothedSparseCategoricalCrossentropy
    return SmoothedSparseCategoricalCrossentropy


def build_static_model(
    class_names: Sequence[str],
    image_size: tuple[int, int] = IMAGE_SIZE,
    learning_rate: float = 1e-3,
    train_base: bool = False,
):
    """Build MobileNetV2 transfer learning with an explicit class ordering."""
    names = list(class_names)
    if len(names) < 2 or len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError("At least two unique, non-empty class names are required")
    keras = _keras()
    base = keras.applications.MobileNetV2(
        input_shape=(*image_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = train_base
    inputs = keras.Input(shape=(*image_size, 3), name="image")
    x = keras.layers.RandomRotation(0.05)(inputs)
    x = keras.layers.RandomTranslation(0.05, 0.05)(x)
    x = keras.layers.RandomZoom(0.1)(x)
    x = keras.layers.RandomContrast(0.1)(x)
    x = base(x, training=train_base)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(len(names), activation="softmax", name="sign")(x)
    model = keras.Model(inputs, outputs, name="signtrans_mobilenet_v2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_sequence_model(
    class_names: Sequence[str],
    sequence_length: int,
    feature_dim: int = 63,
    learning_rate: float = 1e-3,
    architecture: str = "bilstm",
    label_smoothing: float = 0.0,
    conv_frontend: bool = False,
):
    """Build a recurrent classifier for normalized temporal landmark sequences only.

    ``architecture`` is one of ``lstm``, ``gru``, or ``bilstm``. A Transformer is
    intentionally not the default: the landmark clips are short (~30 frames) and
    the dataset is modest, where recurrent models are stronger.

    ``conv_frontend`` prepends two depthwise-friendly Conv1D layers so the
    recurrent layers receive locally-filtered motion patterns, which helps when
    per-class data is sparse. ``label_smoothing`` is passed to the
    cross-entropy loss to keep confidence calibration honest.
    """
    names = list(class_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("At least two unique class names are required")
    if sequence_length < 2 or feature_dim < 1:
        raise ValueError("Invalid sequence shape")
    choice = architecture.strip().lower()
    if choice not in {"lstm", "gru", "bilstm", "attention"}:
        raise ValueError("architecture must be one of: lstm, gru, bilstm, attention")
    keras = _keras()
    inputs = keras.Input(shape=(sequence_length, feature_dim), name="landmarks")
    if conv_frontend:
        # A bias-free convolutional front-end keeps fully-padded frames at
        # exactly zero, so a subsequent Masking layer still recovers the frame
        # validity mask. (Using a Lambda here would break safe model deserialization.)
        x = keras.layers.Conv1D(64, 5, padding="same", activation="relu", use_bias=False, name="conv_front_1")(inputs)
        x = keras.layers.Conv1D(32, 5, padding="same", activation="relu", use_bias=False, name="conv_front_2")(x)
        x = keras.layers.Masking(mask_value=0.0)(x)
    else:
        x = keras.layers.Masking(mask_value=0.0)(inputs)
    if choice == "lstm":
        x = keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
        x = keras.layers.LSTM(64, dropout=0.2)(x)
    elif choice == "gru":
        x = keras.layers.GRU(128, return_sequences=True, dropout=0.2)(x)
        x = keras.layers.GRU(64, dropout=0.2)(x)
    elif choice == "attention":
        # Recurrent encoder + self-attention + temporal average pooling so the
        # classifier can weight the most informative frames instead of relying
        # only on the final hidden state.
        x = keras.layers.Bidirectional(keras.layers.LSTM(96, return_sequences=True, dropout=0.2))(x)
        x = keras.layers.LayerNormalization()(x)
        attended = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
        x = keras.layers.Add()([x, attended])
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
    else:
        x = keras.layers.Bidirectional(keras.layers.LSTM(96, return_sequences=True, dropout=0.2))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(48, dropout=0.2))(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(len(names), activation="softmax", name="sign")(x)
    model = keras.Model(inputs, outputs, name=f"signtrans_landmark_{choice}")
    loss = "sparse_categorical_crossentropy"
    if label_smoothing:
        loss = smoothed_sparse_crossentropy_class()(label_smoothing=float(label_smoothing))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_dynamic_model(
    class_names: Sequence[str],
    sequence_length: int,
    feature_dim: int = 63,
    learning_rate: float = 1e-3,
):
    """Backward-compatible LSTM factory used by the generic dynamic trainer."""
    return build_sequence_model(
        class_names,
        sequence_length,
        feature_dim=feature_dim,
        learning_rate=learning_rate,
        architecture="lstm",
    )


def load_keras_model(path):
    """Load a real Keras artifact, importing TensorFlow only when needed."""
    try:
        # Register the custom label-smoothing loss so models that used it can be
        # deserialized in safe mode.
        smoothed_sparse_crossentropy_class()
    except Exception:  # noqa: BLE001 - older artifacts don't need the custom loss
        pass
    return _keras().models.load_model(path)
