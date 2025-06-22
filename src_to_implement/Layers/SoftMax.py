class SoftMax:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        # Numerical stability: subtract max
        exp = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, error_tensor):
        # Placeholder for backward pass (you can implement this properly later)
        return error_tensor