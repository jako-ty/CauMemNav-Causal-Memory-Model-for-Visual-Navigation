#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch import distributed as distrib
from torch import nn as nn

##############类定义与初始化######################
class RunningMeanAndVar(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        assert n_channels > 0
        self.register_buffer("_mean", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_var", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_count", torch.zeros(()))
        self._mean: torch.Tensor = self._mean
        self._var: torch.Tensor = self._var
        self._count: torch.Tensor = self._count

    ##############前向传播######################
    def forward(self, x: Tensor) -> Tensor:
        #训练模式下统计量更新
        if self.training:
            n = x.size(0)
            # We will need to do reductions (mean) over the channel dimension,
            # so moving channels to the first dimension and then flattening
            # will make those faster.  Further, it makes things more numerically stable
            # for fp16 since it is done in a single reduction call instead of
            # multiple
            x_channels_first = (
                x.transpose(1, 0).contiguous().view(x.size(1), -1)
            )
            new_mean = x_channels_first.mean(-1, keepdim=True)
            new_count = torch.full_like(self._count, n)

            #支持分布式
            if distrib.is_initialized():
                distrib.all_reduce(new_mean)
                distrib.all_reduce(new_count)
                new_mean /= distrib.get_world_size()


            new_var = (
                (x_channels_first - new_mean).pow(2).mean(dim=-1, keepdim=True)
            )

            if distrib.is_initialized():
                distrib.all_reduce(new_var)
                new_var /= distrib.get_world_size()

            #计算当前批次方差
            new_mean = new_mean.view(1, -1, 1, 1)
            new_var = new_var.view(1, -1, 1, 1)

            #合并新旧统计量
            m_a = self._var * (self._count)
            m_b = new_var * (new_count)
            M2 = (
                m_a
                + m_b
                + (new_mean - self._mean).pow(2)
                * self._count
                * new_count
                / (self._count + new_count)
            )

            #更新运行方差及均值
            self._var = M2 / (self._count + new_count)
            self._mean = (self._count * self._mean + new_count * new_mean) / (
                self._count + new_count
            )

            self._count += new_count

        #标准输入化
        inv_stdev = torch.rsqrt(
            torch.max(self._var, torch.full_like(self._var, 1e-2))
        )
        # This is the same as
        # (x - self._mean) * inv_stdev but is faster since it can
        # make use of addcmul and is more numerically stable in fp16
        return torch.addcmul(-self._mean * inv_stdev, x, inv_stdev)
