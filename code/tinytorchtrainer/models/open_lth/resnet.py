# Modified from https://raw.githubusercontent.com/facebookresearch/open_lth/main/models/cifar_resnet.py
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
from functools import partial
import re

class SafeDivide(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        div = a / b
        # Replace non-finite values (like NaN or Inf) with 0
        result = torch.where(torch.isfinite(div), div, torch.tensor(0.0))
        # Save inputs for backward pass
        ctx.save_for_backward(a, b)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors  # Retrieve saved tensors
        grad_a = grad_output / b  # Derivative w.r.t 'a'
        grad_b = -grad_output * a / (b * b)  # Derivative w.r.t 'b'

        valid = torch.isfinite(grad_a) & torch.isfinite(grad_b)

        grad_a_safe = torch.where(valid, grad_a, torch.tensor(0.0))
        grad_b_safe = torch.where(valid, grad_b, torch.tensor(0.0))
        
        return grad_a_safe, grad_b_safe

def safe_divide(a, b):
    return SafeDivide.apply(a, b)

# Reciprocate the real pole if it is outside the unit circle.
def stabilize_1st_order_predictor(a_1):
    return torch.where(torch.abs(a_1) > 1.0, safe_divide(torch.tensor(1.0), a_1), a_1)

# Reciprocate the complex conjugate pair of poles if they are outside the unit circle.
def stabilize_2nd_order_predictor_with_complex_poles(coefs):
    condition = coefs[1] < -1.0
    stabilized = torch.stack((safe_divide(-coefs[0], coefs[1]), safe_divide(torch.tensor(1.0), coefs[1])))
    return torch.where(condition, stabilized, coefs)

# Reciprocate each real pole if it is outside the unit circle.
def stabilize_2nd_order_predictor_with_real_poles(coefs):
    # Find poles
    s = torch.sqrt(coefs[0]**2 + 4.0 * coefs[1]) * 0.5
    poles = coefs[0] * 0.5 + torch.stack((s, -s))
    
    # Reciprocate if needed
    poles = torch.where(torch.abs(poles) > 1.0, safe_divide(torch.tensor(1.0), poles), poles)
    
    # Return coefficients
    return torch.stack((poles[0] + poles[1], -poles[0] * poles[1]))

# Stabilize a 2nd order recurrent 1D predictor.
def stabilize_2nd_order_predictor(coefs):
    # Check if we have complex conjugate poles
    condition = coefs[0] * coefs[0] + 4.0 * coefs[1] < 0.0
    
    # Stabilize based on condition
    stabilized_complex = stabilize_2nd_order_predictor_with_complex_poles(coefs)
    stabilized_real = stabilize_2nd_order_predictor_with_real_poles(coefs)
    
    return torch.where(condition, stabilized_complex, stabilized_real)

def lp1x1cs(input: torch.Tensor, num_pad: int=1):
    mean = torch.mean(input, dim=(2, 3), keepdim=True)
    input = input - mean
    r_11 = torch.mean(input[:, :, :, :-1] * input[:, :, :, :-1], dim=(2, 3), keepdim=True)
    r_01 = torch.mean(input[:, :, :, 1:] * input[:, :, :, :-1], dim=(2, 3), keepdim=True)
    l_11 = torch.mean(input[:, :, :, 1:] * input[:, :, :, 1:], dim=(2, 3), keepdim=True)
    l_01 = torch.mean(input[:, :, :, :-1] * input[:, :, :, 1:], dim=(2, 3), keepdim=True)
    b_11 = torch.mean(input[:, :, :-1, :] * input[:, :, :-1, :], dim=(2, 3), keepdim=True)
    b_01 = torch.mean(input[:, :, 1:, :] * input[:, :, :-1, :], dim=(2, 3), keepdim=True)
    t_11 = torch.mean(input[:, :, 1:, :] * input[:, :, 1:, :], dim=(2, 3), keepdim=True)
    t_01 = torch.mean(input[:, :, :-1, :] * input[:, :, 1:, :], dim=(2, 3), keepdim=True)
    ra_1 = safe_divide(r_01, r_11)
    la_1 = safe_divide(l_01, l_11)
    ba_1 = safe_divide(b_01, b_11)
    ta_1 = safe_divide(t_01, t_11)
    ra_1 = stabilize_1st_order_predictor(ra_1)
    la_1 = stabilize_1st_order_predictor(la_1)
    ba_1 = stabilize_1st_order_predictor(ba_1)
    ta_1 = stabilize_1st_order_predictor(ta_1)

    padded = input
    while num_pad > 0:
        padded = torch.cat([
            padded[:, :, :1, :]*ta_1,
            padded[:, :, :, :],
            padded[:, :, -1:, :]*ba_1
        ], dim=2)
        padded = torch.cat([
            padded[:, :, :, :1]*la_1,
            padded[:, :, :, :],
            padded[:, :, :, -1:]*ra_1
        ], dim=3)
        num_pad -= 1

    return padded + mean

def lp2x1cs(input: torch.Tensor, num_pad: int=1):
    mean = torch.mean(input, dim=(2, 3), keepdim=True)
    input = (input - mean)
    r_11 = torch.mean(input[:, :, :,1:-1] * input[:, :, :,1:-1], dim=(2, 3), keepdim=True)
    r_22 = torch.mean(input[:, :, :,0:-2] * input[:, :, :,0:-2], dim=(2, 3), keepdim=True)
    r_01 = torch.mean(input[:, :, :,2:] * input[:, :, :,1:-1], dim=(2, 3), keepdim=True)
    r_12 = torch.mean(input[:, :, :,1:-1] * input[:, :, :,0:-2], dim=(2, 3), keepdim=True)
    r_02 = torch.mean(input[:, :, :,2:] * input[:, :, :,0:-2], dim=(2, 3), keepdim=True)

    l_11 = torch.mean(input[:, :, :,1:-1] * input[:, :, :,1:-1], dim=(2, 3), keepdim=True)
    l_22 = torch.mean(input[:, :, :,2:] * input[:, :, :,2:], dim=(2, 3), keepdim=True)
    l_01 = torch.mean(input[:, :, :,:-2] * input[:, :, :,1:-1], dim=(2, 3), keepdim=True)
    l_12 = torch.mean(input[:, :, :,1:-1] * input[:, :, :,2:], dim=(2, 3), keepdim=True)
    l_02 = torch.mean(input[:, :, :,:-2] * input[:, :, :,2:], dim=(2, 3), keepdim=True)

    b_11 = torch.mean(input[:, :, 1:-1,:] * input[:, :, 1:-1,:], dim=(2, 3), keepdim=True)
    b_22 = torch.mean(input[:, :, 0:-2,:] * input[:, :, 0:-2,:], dim=(2, 3), keepdim=True)
    b_01 = torch.mean(input[:, :, 2:,:] * input[:, :, 1:-1,:], dim=(2, 3), keepdim=True)
    b_12 = torch.mean(input[:, :, 1:-1,:] * input[:, :, 0:-2,:], dim=(2, 3), keepdim=True)
    b_02 = torch.mean(input[:, :, 2:,:] * input[:, :, 0:-2,:], dim=(2, 3), keepdim=True)

    t_11 = torch.mean(input[:, :, 1:-1,:] * input[:, :, 1:-1,:], dim=(2, 3), keepdim=True)
    t_22 = torch.mean(input[:, :, 2:,:] * input[:, :, 2:,:], dim=(2, 3), keepdim=True)
    t_01 = torch.mean(input[:, :, :-2,:] * input[:, :, 1:-1,:], dim=(2, 3), keepdim=True)
    t_12 = torch.mean(input[:, :, 1:-1,:] * input[:, :, 2:,:], dim=(2, 3), keepdim=True)
    t_02 = torch.mean(input[:, :, :-2,:] * input[:, :, 2:,:], dim=(2, 3), keepdim=True)

    ra_1 = safe_divide(r_01*r_22 - r_02*r_12, r_11*r_22 - r_12*r_12)
    ra_2 = safe_divide(r_02*r_11 - r_01*r_12, r_11*r_22 - r_12*r_12)

    la_1 = safe_divide(l_01*l_22 - l_02*l_12, l_11*l_22 - l_12*l_12)
    la_2 = safe_divide(l_02*l_11 - l_01*l_12, l_11*l_22 - l_12*l_12)

    ba_1 = safe_divide(b_01*b_22 - b_02*b_12, b_11*b_22 - b_12*b_12)
    ba_2 = safe_divide(b_02*b_11 - b_01*b_12, b_11*b_22 - b_12*b_12)

    ta_1 = safe_divide(t_01*t_22 - t_02*t_12, t_11*t_22 - t_12*t_12)
    ta_2 = safe_divide(t_02*t_11 - t_01*t_12, t_11*t_22 - t_12*t_12)
    
    r_coefs = stabilize_2nd_order_predictor(torch.stack((ra_1, ra_2)))
    ra_1, ra_2 = (r_coefs[0], r_coefs[1])
    l_coefs = stabilize_2nd_order_predictor(torch.stack((la_1, la_2)))
    la_1, la_2 = (l_coefs[0], l_coefs[1])
    b_coefs = stabilize_2nd_order_predictor(torch.stack((ba_1, ba_2)))
    ba_1, ba_2 = (b_coefs[0], b_coefs[1])
    t_coefs = stabilize_2nd_order_predictor(torch.stack((ta_1, ta_2)))
    ta_1, ta_2 = (t_coefs[0], t_coefs[1])

    padded = input
    while num_pad > 0:
        padded = torch.cat([
            (padded[:, :, :1, :]*ta_1 + padded[:, :, 1:2, :]*ta_2).to(padded.dtype),
            padded[:, :, :, :],
            (padded[:, :, -1:, :]*ba_1 + padded[:, :, -2:-1, :]*ba_2).to(padded.dtype)
        ], dim=2)
        padded = torch.cat([
            (padded[:, :, :, :1]*la_1 + padded[:, :, :, 1:2]*la_2).to(padded.dtype),
            padded[:, :, :, :],
            (padded[:, :, :, -1:]*ra_1 + padded[:, :, :, -2:-1]*ra_2).to(padded.dtype)
        ], dim=3)
        num_pad -= 1

    return padded + mean

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        @staticmethod
        def make_conv(f_in: int, f_out: int, stride: int, conv_type: str, kernel_size:int=3, force_no_padding: bool=False, padding_mode: str="zeros"):
            assert padding_mode in ["zeros", "reflect", "replicate", "circular", "lp1x1cs", "lp2x1cs"], f"Invalid padding mode {padding_mode}."

            padding = (kernel_size // 2)
            if force_no_padding:
                padding = 0
            
            if not conv_type:
                if padding_mode == "lp1x1cs":
                    return nn.Sequential(
                        Lambda(lambda input: lp1x1cs(input, padding)),
                        nn.Conv2d(f_in, f_out, kernel_size=kernel_size, stride=stride, bias=False)
                    )
                if padding_mode == "lp2x1cs":
                    return nn.Sequential(
                        Lambda(lambda input: lp2x1cs(input, padding)),
                        nn.Conv2d(f_in, f_out, kernel_size=kernel_size, stride=stride, bias=False)
                    )
                else:
                    return nn.Conv2d(f_in, f_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False)
            elif conv_type == "depthwise_separable":
                if padding_mode == "lp1x1cs":
                    return nn.Sequential(
                        Lambda(lambda input: lp1x1cs(input, padding)),
                        nn.Conv2d(f_in, f_in, kernel_size=kernel_size, stride=stride, bias=False, groups=f_in),
                        nn.Conv2d(f_in, f_out, kernel_size=1, bias=False),
                    )
                elif padding_mode == "lp2x1cs":
                    return nn.Sequential(
                        Lambda(lambda input: lp2x1cs(input, padding)),
                        nn.Conv2d(f_in, f_in, kernel_size=kernel_size, stride=stride, bias=False, groups=f_in),
                        nn.Conv2d(f_in, f_out, kernel_size=1, bias=False),
                    )
                else:
                    return nn.Sequential(
                        nn.Conv2d(f_in, f_in, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=False, groups=f_in),
                        nn.Conv2d(f_in, f_out, kernel_size=1, bias=False),
                    )
            else:
                raise ValueError(f"Invalid conv_type {conv_type}.")


        def __init__(self, f_in: int, f_out: int, activation_fn, downsample=False, **conv_args):
            super().__init__()

            stride = 2 if downsample else 1
            self.downsample = downsample
            
            self.conv1 = ResNet.Block.make_conv(f_in, f_out, stride, **conv_args)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = ResNet.Block.make_conv(f_out, f_out, 1, **conv_args)
            self.bn2 = nn.BatchNorm2d(f_out)

            self.force_no_padding = conv_args.get("force_no_padding", False)

            self.activation = activation_fn()

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = x
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.activation(out)
            
            out = self.conv2(out)
            out = self.bn2(out)

            if self.force_no_padding:
              if self.downsample:
                out = nn.functional.pad(out, (2, 1, 2, 1))
              else:
                out = nn.functional.pad(out, (2, 2, 2, 2))

            out += self.shortcut(x)
            return self.activation(out)

    def __init__(self, plan, in_channels=3, num_classes=10, activation_fn=None, **conv_args):
        super().__init__()

        if activation_fn is None:
            activation_fn = partial(nn.ReLU, inplace=True)

        # Initial convolution.
        current_filters = plan[0][0]

        self.conv = ResNet.Block.make_conv(in_channels, current_filters, stride=1, **conv_args)
        self.bn = nn.BatchNorm2d(current_filters)
        self.activation = activation_fn()

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(ResNet.Block(current_filters, filters, activation_fn=activation_fn, downsample=downsample, **conv_args))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], num_classes)

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = self.bn(out)
        out = self.activation(out)
        out = self.blocks(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @property
    def output_layer_names(self):
        return ["fc.weight", "fc.bias"]

    @staticmethod
    def get_model_from_name(name, **kwargs):
        use_1x1_stem = False
        kernel_size = 3
        conv_type = None
        padding_mode = "zeros"
        force_no_padding = False

        if "_dw" in name:
            conv_type = "depthwise_separable"
            name = name.replace("_dw", "")
          
        if "_1x1stem" in name:
            use_1x1_stem = True
            name = name.replace("_1x1stem", "")
            
        exp_match = re.search(r"_k[0-9]+", name)
        if exp_match:
            kernel_size = int(exp_match.group(0)[2:])
            name = name.replace(exp_match.group(0), "")

        exp_match = re.search(r"_padding(.*)", name)
        if exp_match:
            padding_mode = exp_match.group(1)
            name = name.replace(exp_match.group(0), "")

        if "_nopad" in name:
            force_no_padding = True
            name = name.replace("_nopad", "")

        name = name.split("_")
        
        assert len(name) <= 3, f"Extra args not understood {name}"

        W = 16 if len(name) == 2 else int(name[2])
        D = int(name[1])
        if (D - 2) % 3 != 0:
            raise ValueError("Invalid ResNet depth: {}".format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        resnet = ResNet(plan, conv_type=conv_type, kernel_size=kernel_size, padding_mode=padding_mode, force_no_padding=force_no_padding, **kwargs)

        # must be first, before any other replace!
        if use_1x1_stem:
            resnet.conv = nn.Conv2d(in_channels=resnet.conv.in_channels, out_channels=resnet.conv.out_channels, kernel_size=1, bias=False)

        return resnet