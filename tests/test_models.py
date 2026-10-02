import numpy as np
import pytest

from mnist_study.data import to_float
from mnist_study.models import build, configure


def test_parameter_counts_match_hand_calculation():
    # MLP: 784*128+128 + 128*64+64 + 64*10+10
    assert build("mlp").count_params() == 100_480 + 8_256 + 650
    # CNN: conv 3*3*1*32+32, conv 3*3*32*64+64, dense 5*5*64*128+128, head 128*10+10
    assert build("cnn").count_params() == 320 + 18_496 + 204_928 + 1_290


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build("resnet")


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_models_learn_real_digits(small_mnist, name):
    x_train, y_train, x_test, y_test = small_mnist
    configure(seed=0)
    model = build(name)
    model.fit(to_float(x_train), y_train, epochs=8, batch_size=32, verbose=0)
    acc = float(np.mean(model.predict(to_float(x_test), verbose=0).argmax(1) == y_test))
    # 600 training digits are enough to be far above the 10% chance level.
    assert acc > 0.75


def test_same_seed_gives_identical_predictions(small_mnist):
    x_train, y_train, x_test, _ = small_mnist
    outs = []
    for _ in range(2):
        configure(seed=3)
        model = build("mlp")
        model.fit(to_float(x_train[:200]), y_train[:200], epochs=1, batch_size=32, verbose=0)
        outs.append(model.predict(to_float(x_test[:20]), verbose=0))
    np.testing.assert_array_equal(outs[0], outs[1])
