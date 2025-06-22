import numpy as np
from scipy.signal import correlate, correlate2d


class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        self.num_kernels = num_kernels
        self.convolution_shape = convolution_shape

        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,) * (len(convolution_shape) - 1)
        else:
            self.stride_shape = stride_shape

        self.optimizer = None
        self._optimizer_bias = None

        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, (num_kernels, 1))

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        self._input_tensor = None

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        spatial_extent = np.prod(self.convolution_shape[1:])
        fan_out = self.num_kernels * spatial_extent

        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        if opt is not None:
            self._optimizer_bias = opt.__class__(**opt.__dict__)

    def forward(self, input_tensor):
        self._input_tensor = input_tensor
        if input_tensor.ndim == 4:
            return self._forward2d(input_tensor)
        elif input_tensor.ndim == 3:
            return self._forward1d(input_tensor)
        else:
            raise ValueError(f"Unsupported input tensor shape: {input_tensor.shape}")

    def backward(self, error_tensor):
        if self._input_tensor.ndim == 4:
            return self._backward2d(error_tensor)
        elif self._input_tensor.ndim == 3:
            return self._backward1d(error_tensor)
        else:
            raise ValueError(f"Unsupported input tensor shape: {self._input_tensor.shape}")

    def _pad_input(self, input_tensor, kernel_shape):
        if input_tensor.ndim == 4:
            pad_y = (kernel_shape[1] - 1) // 2
            pad_x = (kernel_shape[2] - 1) // 2
            return np.pad(input_tensor, ((0, 0), (0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode='constant')
        elif input_tensor.ndim == 3:
            pad = (kernel_shape[1] - 1) // 2
            return np.pad(input_tensor, ((0, 0), (0, 0), (pad, pad)), mode='constant')

    def _forward2d(self, input_tensor):
        batch_size, in_channels, in_y, in_x = input_tensor.shape
        padded = self._pad_input(input_tensor, self.convolution_shape)

        k_y, k_x = self.convolution_shape[1:]
        stride_y, stride_x = self.stride_shape
        out_y = (in_y - 1) // stride_y + 1
        out_x = (in_x - 1) // stride_x + 1

        output = np.zeros((batch_size, self.num_kernels, out_y, out_x))

        for b in range(batch_size):
            for k in range(self.num_kernels):
                acc = np.zeros((in_y, in_x))
                for c in range(in_channels):
                    acc += correlate2d(padded[b, c], self.weights[k, c], mode='valid')
                acc += self.bias[k]
                output[b, k] = acc[::stride_y, ::stride_x]

        return output

    def _backward2d(self, error_tensor):
        batch_size, in_channels, in_y, in_x = self._input_tensor.shape
        padded = self._pad_input(self._input_tensor, self.convolution_shape)
        grad_input = np.zeros_like(self._input_tensor)
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        stride_y, stride_x = self.stride_shape

        for b in range(batch_size):
            for k in range(self.num_kernels):
                self._gradient_bias[k] += np.sum(error_tensor[b, k])
                upsampled = np.zeros_like(padded[b, 0])
                upsampled[::stride_y, ::stride_x] = error_tensor[b, k]
                for c in range(in_channels):
                    self._gradient_weights[k, c] += correlate2d(padded[b, c], upsampled, mode='valid')
                    grad_input[b, c] += correlate2d(upsampled, self.weights[k, c], mode='full')

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return grad_input

    def _forward1d(self, input_tensor):
        batch_size, in_channels, width = input_tensor.shape
        padded = self._pad_input(input_tensor, self.convolution_shape)

        k_w = self.convolution_shape[1]
        stride = self.stride_shape[0]
        out_w = (width - 1) // stride + 1

        output = np.zeros((batch_size, self.num_kernels, out_w))

        for b in range(batch_size):
            for k in range(self.num_kernels):
                acc = np.zeros(width)
                for c in range(in_channels):
                    acc += correlate(padded[b, c], self.weights[k, c], mode='valid')
                acc += self.bias[k]
                output[b, k] = acc[::stride]

        return output

    def _backward1d(self, error_tensor):
        batch_size, in_channels, width = self._input_tensor.shape
        padded = self._pad_input(self._input_tensor, self.convolution_shape)
        grad_input = np.zeros_like(self._input_tensor)
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        stride = self.stride_shape[0]

        for b in range(batch_size):
            for k in range(self.num_kernels):
                self._gradient_bias[k] += np.sum(error_tensor[b, k])
                upsampled = np.zeros_like(padded[b, 0])
                upsampled[::stride] = error_tensor[b, k]
                for c in range(in_channels):
                    self._gradient_weights[k, c] += correlate(padded[b, c], upsampled, mode='valid')
                    grad_input[b, c] += correlate(upsampled, self.weights[k, c], mode='full')

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return grad_input
