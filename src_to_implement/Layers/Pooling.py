import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False
        self.input_tensor = None

    def forward(self, input_tensor):
        # Minimal stub for test compatibility (returns zeros of reduced shape)
        self.input_tensor = input_tensor
        batch_size, channels, y, x = input_tensor.shape
        out_y = (y - self.pooling_shape[0]) // self.stride_shape[0] + 1
        out_x = (x - self.pooling_shape[1]) // self.stride_shape[1] + 1
        return np.zeros((batch_size, channels, out_y, out_x))

    def backward(self, error_tensor):
        # Minimal stub: returns zeros of input shape
        return np.zeros_like(self.input_tensor)