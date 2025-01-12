import torch


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.kernel_size = 3
        self.resnetBlock = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, stride=stride),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(),
        )
        self.conv2d_for_input = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=(1, 1), stride=stride)
        self.batchnorm2d_for_input = torch.nn.BatchNorm2d(num_features=out_channels)

        # not needed anymore, default in pytorch is already He for weights in convolution layers
        # torch.nn.init.kaiming_uniform_(self.resnetBlock[0].weight, mode='fan_in')
        # torch.nn.init.kaiming_uniform_(self.conv2d_for_input.weight, mode='fan_in')

        # not needed anymore, default in pytorch is already zero initialization for biases
        # torch.nn.init.zeros_(self.resnetBlock[0].bias)
        # torch.nn.init.zeros_(self.conv2d_for_input.bias)

    def forward(self, input_tensor):
        modified_input = self.conv2d_for_input(input_tensor)
        modified_input = self.batchnorm2d_for_input(modified_input)

        padded_input_tensor = pad_input_tensor(input_tensor, self.kernel_size)
        resBlock_result = self.resnetBlock(padded_input_tensor)
        return resBlock_result + modified_input


def pad_input_tensor(tensor, kernel_size):
    # print("Shape before padding: ", tensor.shape)
    input_is_1d = True if len(tensor.shape) == 3 else False
    padding_size_y_before = kernel_size // 2
    padding_size_x_before = kernel_size // 2

    # Use different padding for even/uneven Kernels

    if kernel_size % 2 == 0:
        padding_size_y_after = kernel_size // 2 - 1
    else:
        padding_size_y_after = padding_size_y_before

    if kernel_size % 2 == 0:
        padding_size_x_after = kernel_size // 2 - 1
    else:
        padding_size_x_after = padding_size_x_before

    if input_is_1d:
        padded_input_tensor = torch.nn.functional.pad(tensor,
                                                      [0, 0, 0, 0, padding_size_y_before, padding_size_y_after],
                                                      "constant", 0)
    else:
        padded_input_tensor = torch.nn.functional.pad(tensor,
                                                      [padding_size_y_before, padding_size_y_after,
                                                       padding_size_x_before, padding_size_x_after, 0, 0, 0, 0, ],
                                                      "constant", 0)
    #  left, right, top, bottom, front, back
    # print("Shape after padding: ", padded_input_tensor.shape)

    return padded_input_tensor
