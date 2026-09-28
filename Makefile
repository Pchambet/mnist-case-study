PY=python3

.PHONY: mlp cnn clean

mlp:
	$(PY) train_mlp.py

clean:
	rm -rf outputs __pycache__ .ipynb_checkpoints
