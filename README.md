# MNIST Case Study – Deep Learning From Scratch

A hands-on, honest, and didactic deep learning project built from the ground up.

> “No shortcuts. No illusions. Just code, clarity, and comprehension.”

---

## Overview

This repository explores the famous **MNIST handwritten digit dataset** through a complete and transparent deep learning workflow — starting from scratch.

Unlike typical libraries that hide complexity, this notebook walks through:
- manual implementation of a neuron
- sigmoid activation and log-loss
- forward and backward propagation
- simple network training and evaluation
- visual decision boundaries

Everything is explained, coded, and visualized.

---

## How to Run

### Requirements

You'll need:
- Python ≥ 3.8
- `numpy`, `matplotlib`, and either `tensorflow` or `torch`

Install required packages:

```
pip install numpy matplotlib tensorflow
```

### Run the notebook

You can run it locally with:

```
jupyter notebook mnist.ipynb
```

Or open it directly in Google Colab:

[Open in Colab](https://colab.research.google.com/github/Pchambet/mnist-case-study/blob/main/mnist.ipynb)

---

## Dataset Note

⚠️ **The MNIST dataset is not stored in this repository.**  
It will be downloaded automatically when running the notebook using:

```python
from tensorflow.keras.datasets import mnist
```

---

## Philosophy

This notebook is part of **WIL™ – Wide-Range Ideas Laboratory**, an independent educational space created by Pierre Chambet in 2025.

> At WIL, we believe in deep learning — both technically and humanly.

We reject black-box shortcuts and aim to build intuition, one line at a time.

---

## Author

**Pierre Chambet**  
Dual Master Student at Télécom SudParis & École Polytechnique  
Research intern in Computational Systems Biology (NAIST, Japan)  
Founder of WIL™ – Wide-Range Ideas Laboratory  
[LinkedIn](https://www.linkedin.com/in/pierrechambet)

---

## License

This project is released under the MIT License.  
Feel free to use, remix, or extend it.