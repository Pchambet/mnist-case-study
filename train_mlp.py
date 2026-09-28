import os
import random
import numpy as np

# Prefer TensorFlow on Apple Silicon per requirements.txt
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # Flatten for MLP
    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))
    return (x_train, y_train), (x_test, y_test)


def build_mlp(input_dim: int = 784, num_classes: int = 10) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def main():
    set_seed(42)
    (x_train, y_train), (x_test, y_test) = load_mnist()

    model = build_mlp()

    os.makedirs("outputs", exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="outputs/mlp_mnist.keras", save_best_only=True, monitor="val_accuracy"
        ),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print({"test_accuracy": float(test_acc), "test_loss": float(test_loss)})

    # Save training curves
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.title("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="train")
        plt.plot(history.history["val_accuracy"], label="val")
        plt.title("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig("outputs/mlp_training_curves.png", dpi=150)
        plt.close()
    except Exception as e:  # Optional plotting
        print(f"Warning: could not save curves: {e}")


if __name__ == "__main__":
    main()
