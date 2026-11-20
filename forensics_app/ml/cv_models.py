"""Set 8 CIFAR-10 model architectures."""

from __future__ import annotations

from typing import Callable

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

FEATURE_LAYER_NAMES = {
    "custom_cnn": "features",
    "inceptionv3_transfer": "feature_vector",
}


def build_custom_cnn(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
) -> tf.keras.Model:
    """Build the compact 3-block CNN (about 89k parameters)."""
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    for index, filters in enumerate((32, 64, 96), start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{index}")(x)
        x = layers.BatchNormalization(name=f"bn{index}")(x)
        x = layers.ReLU(name=f"relu{index}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{index}")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="features")(x)
    x = layers.Dropout(0.3, name="dropout_features")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="custom_cnn")


def build_inceptionv3_transfer(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Build the Colab-compatible InceptionV3 model.

    Inputs are RGB float pixels in [0, 1]. The Rescaling layer matches the
    Colab model's 2*x-1 preprocessing before its frozen InceptionV3 backbone.
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = layers.Resizing(75, 75, interpolation="bilinear", name="resize_75")(inputs)
    x = layers.Rescaling(2.0, offset=-1.0, name="inception_preprocess")(x)
    base_model = InceptionV3(
        include_top=False,
        weights=weights,
        input_shape=(75, 75, 3),
        pooling="avg",
        name="inception_v3",
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(256, activation="relu", name="feature_vector")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="inceptionv3_cifar10")


BUILDERS: dict[str, Callable[..., tf.keras.Model]] = {
    "custom_cnn": build_custom_cnn,
    "inceptionv3_transfer": build_inceptionv3_transfer,
}

__all__ = [
    "CLASS_NAMES",
    "FEATURE_LAYER_NAMES",
    "build_custom_cnn",
    "build_inceptionv3_transfer",
    "BUILDERS",
]
