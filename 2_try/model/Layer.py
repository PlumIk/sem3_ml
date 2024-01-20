import numpy as np

# Инит весов
def _init_weights_data(input_size, output_size):
    weights = np.random.randn(input_size, output_size) / 10.
    weights_grads = np.empty_like(weights)
    return weights, weights_grads

# Смещение
def _init_bias_data(bias, layer_size):
    bias_values = None
    bias_grads = None
    if bias:
        bias_values = np.random.randn(layer_size) / 10.
        bias_grads = np.empty_like(bias_values)
    return bias_values, bias_grads


class Layer:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __init__(self, input_size, output_size, bias):
        self._weights, self._weights_grads = _init_weights_data(input_size, output_size)
        self._layer_input = None
        self._biases, self._biases_grads = _init_bias_data(bias, output_size)

    def _is_bias(self):
        return self._biases is not None

    # Результат слоя
    def forward(self, x):
        self._layer_input = x
        linear_value = self._layer_input @ self._weights
        if self._is_bias():
            linear_value += self._biases
        return linear_value

    # Градиенты лосс по весам
    def backward(self, x):
        output_grad = x
        if self._is_bias():
            self._biases_grads = (np.ones(shape=(1, x.shape[0])) @ output_grad).squeeze()
        self._weights_grads = self._layer_input.T @ output_grad
        return output_grad @ self._weights.T

    # обновить
    def step(self, lr):
        self._weights -= self._weights_grads * lr
        if self._biases is not None:
            self._biases -= self._biases_grads * lr
