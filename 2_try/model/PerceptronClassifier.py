from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pickle
from functools import reduce


class SoftMaxCrossEntropy:
    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)

    def forward(self, y: np.ndarray, ygt: np.ndarray) -> np.ndarray:
        softmax = np.exp(y) / (np.exp(y).sum(axis=1, keepdims=True) + 1e-6)
        return -np.log(softmax[np.arange(y.shape[0]), ygt] + 1e-6).mean()

    def backward(self, y: np.ndarray, ygt: np.ndarray) -> np.ndarray:
        softmax = np.exp(y) / (np.exp(y).sum(axis=1, keepdims=True) + 1e-6)
        softmax[np.arange(y.shape[0]), ygt] -= 1
        return softmax / y.shape[0]


class ReLU:
    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)

    def __init__(self):
        self._activation_value = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._activation_value = np.maximum(0, x)
        return self._activation_value

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x * (self._activation_value > 0).astype(float)


@dataclass
class OutData:
    predictions: np.ndarray
    probas: np.ndarray


class PerceptronClassifier:
    def predict(self, x):
        probas = self.predict_proba(x)
        predictions = np.argmax(probas, axis=1)
        return OutData(predictions, probas)

    def predict_proba(self, x):
        return self.forward(x)

    def __init__(self, layers):
        self._layers = layers

    def store_model(self, store_path):
        with open(store_path, 'wb') as f:
            pickle.dump(self._layers, f)

    @classmethod
    def load_model(cls, store_path):
        with open(store_path, 'rb') as f:
            layers = pickle.load(f)
        return cls(layers)

    def forward(self, x):
        layers = (layer.forward for layer in self._layers)

        res = reduce(lambda layer_input, layer_forward: layer_forward(layer_input),
                     layers,
                     x)
        return res

    def backward(self, output_grad):
        layers_grads = (layer.backward for layer in self._layers[::-1])

        res = reduce(lambda layer_output_grad, layer_backward: layer_backward(layer_output_grad),
                     layers_grads,
                     output_grad)
        return res

    def step(self, lr):
        for layer in self._layers:
            if hasattr(layer, 'step'):
                layer.step(lr)
