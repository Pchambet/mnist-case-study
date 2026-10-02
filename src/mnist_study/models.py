"""The architectures under comparison: an MLP, a CNN and a parameter-matched MLP control.

Both are deliberately small and conventional: the question is what convolution
buys over a dense network of the same order of size, not how far MNIST can be pushed.
The CNN has about twice the MLP's parameters, so a wider MLP with as many parameters
as the CNN is trained as a capacity control.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_NAMES = ("mlp", "cnn")
CONTROL_NAMES = ("mlp_wide",)


def configure(seed: int, threads: int = 3) -> None:
    """Seed every RNG and make CPU kernels deterministic.

    With op determinism on, re-running a seed reproduces the same weights, which
    is what lets the README quote exact numbers.
    """
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # Thread pools can only be sized before TensorFlow initialises them;
        # later calls in the same process keep the first setting.
        pass


def build_mlp(
    num_classes: int = 10, widths: tuple[int, int] = (128, 64), name: str = "mlp"
) -> keras.Model:
    """784-128-64-10: same layer sizes as the walkthrough notebook, plus dropout."""
    return keras.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(widths[0], activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(widths[1], activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name=name,
    )


def build_mlp_wide(num_classes: int = 10) -> keras.Model:
    """784-256-128-10: the same recipe with as many parameters as the CNN (capacity control)."""
    return build_mlp(num_classes, widths=(256, 128), name="mlp_wide")


def build_cnn(num_classes: int = 10) -> keras.Model:
    """Two conv/pool blocks and a dense head: the textbook LeNet-style CNN."""
    return keras.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Reshape((28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn",
    )


def build(name: str, learning_rate: float = 1e-3) -> keras.Model:
    builders = {"mlp": build_mlp, "cnn": build_cnn, "mlp_wide": build_mlp_wide}
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; expected one of {tuple(builders)}")
    model = builders[name]()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
