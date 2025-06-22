class NeuralNetwork:
    def __init__(self, weights_initializer, bias_initializer):
        self.layers = []
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def append_layer(self, layer):
        # If the layer is trainable, initialize it
        if hasattr(layer, 'trainable') and layer.trainable:
            if hasattr(layer, 'initialize'):
                layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self, input_tensor):
        tensor = input_tensor
        for layer in self.layers:
            tensor = layer.forward(tensor)
        return tensor

    def backward(self, error_tensor):
        tensor = error_tensor
        for layer in reversed(self.layers):
            tensor = layer.backward(tensor)
        return tensor