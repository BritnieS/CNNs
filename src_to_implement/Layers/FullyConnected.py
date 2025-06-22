import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        # Weights shape: (input_size + 1, output_size) to include bias as last row
        self.weights = np.random.uniform(0, 1, (self.input_size + 1, self.output_size))
        self._gradient_weights = None
        self.optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        w = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        b = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)
        # Bias is appended as last row of weights
        self.weights = np.vstack([w, b])

    def forward(self, input_tensor):
        # Append 1 for bias to the input tensor
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.reshape(1, -1)
        batch_size = input_tensor.shape[0]
        ones = np.ones((batch_size, 1), dtype=float)
        self.input_tensor = np.hstack([input_tensor, ones])
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        # Compute gradient w.r.t input (excluding bias)
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        grad_input = np.dot(error_tensor, self.weights.T)
        # Remove last column (bias) from gradient
        return grad_input[:, :-1]

    @property
    def gradient_weights(self):
        return self._gradient_weights