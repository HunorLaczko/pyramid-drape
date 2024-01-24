# based on torchvision.models.convnext

from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from utils.convnext_utils import CNBlockConfig


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0.0,
        kernel_size: int = 7,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding='same', groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class ConvNeXtBase(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device: str = 'cuda',
        kernel_size: int = 7,
        is_label: bool = False,
        nr_of_labels: Optional[int] = None,
        input_features: int = 3,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers_list: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        self.stem = Conv2dNormActivation(
                input_features if not is_label else nr_of_labels,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            ).to(device)
        layers_list.append(self.stem)

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob, kernel_size))
                stage_block_id += 1
            layers_list.append(nn.Sequential(*stage).to(device))
            if cnf.out_channels is not None:
                # Downsampling
                layers_list.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    ).to(device)
                )

        self.layers = nn.Sequential(*layers_list).to(device)
        self.act = nn.Tanh().to(device)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.act(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ConvNeXtReverseBase(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device: str = 'cuda',
        use_skip: bool = False,
        is_label: bool = False,
        nr_of_labels: Optional[int] = None,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        self.use_skip = use_skip

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        self.stem = nn.Sequential(
                        nn.ConvTranspose2d(block_setting[0].input_channels, block_setting[0].out_channels, kernel_size=4, stride=4),
                        norm_layer(block_setting[0].out_channels),
                    ).to(device)
        
        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        layers_list: List[nn.Module] = []


        for i, cnf in enumerate(block_setting):
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob, kernel_size))
                stage_block_id += 1
            layers_list.append(nn.Sequential(*stage))

            if i == 0:
                layers_list.append(self.stem)
            else:
                # Upsampling
                layers_list.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.ConvTranspose2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.last = nn.Sequential(
                        Permute([0, 2, 3, 1]),
                        nn.Linear(in_features=block_setting[-1].out_channels, out_features=3 if not is_label else nr_of_labels, bias=True),
                        Permute([0, 3, 1, 2]),
                    ).to(device)
        
        self.layers = nn.Sequential(*layers_list).to(device)
        self.activation = nn.Tanh().to(device) if not is_label else nn.Identity().to(device)

    def _forward_impl(self, x) -> Tensor:
        x = self.layers(x)
        x = self.last(x)
        x = self.activation(x)
        return x

    def forward(self, x) -> Tensor:
        return self._forward_impl(x)



def _convnext(
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    reversed: bool,
    is_label: bool,
    nr_of_labels: Optional[int],
    kernel_size: int,
    input_features: int,
    device='cuda',
    **kwargs: Any,
) -> ConvNeXtBase:

    if reversed:
        model = ConvNeXtReverseBase(block_setting, stochastic_depth_prob=stochastic_depth_prob, is_label=is_label, nr_of_labels=nr_of_labels, kernel_size=kernel_size, device=device, **kwargs)
    else:
        model = ConvNeXtBase(block_setting, stochastic_depth_prob=stochastic_depth_prob, kernel_size=kernel_size, is_label=is_label, nr_of_labels=nr_of_labels, input_features=input_features, device=device, **kwargs)


    return model


class ConvNeXt(nn.Module):
    def __init__(self, block_setting, stochastic_depth_prob=0.0, reversed=False, is_label=False, nr_of_labels=None, kernel_size=7, input_features=3, device='cuda', **kwargs) -> None:
        super().__init__()

        self.model: nn.Module = _convnext(block_setting, stochastic_depth_prob, reversed, is_label, nr_of_labels, kernel_size, input_features=input_features, device=device, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x
        
