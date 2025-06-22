import numpy as np

class Flatten:
    def __init__(self):
        self.trainable = False
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        return input_tensor.reshape(batch_size, -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)