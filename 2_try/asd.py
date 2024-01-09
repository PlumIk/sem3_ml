from Trainer import Trainer
from DataWork import mnist
from model.Layer import Layer
from model.PerceptronClassifier import ReLU
from model.PerceptronClassifier import PerceptronClassifier
from model.PerceptronClassifier import SoftMaxCrossEntropy
from pathlib import Path
import numpy as np

(X_train, y_train, X_test, y_test) = mnist()
layers = [
    Layer(input_size=28 * 28, output_size=32, bias=True),
    ReLU(),
    Layer(input_size=32, output_size=14, bias=True),
    ReLU(),
    Layer(input_size=14, output_size=10, bias=True),
    ReLU()
]
model = PerceptronClassifier(layers)
train_loss = Trainer(model, SoftMaxCrossEntropy()).fit(X_train, y_train, lr=1e-3, epoch_count=1, batch_size=3)
from Trainer import test_classificator

print(test_classificator(model, X_test, y_test))