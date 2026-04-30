# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch

def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)

def pretty_print_module(module, indent_size=4):
    """
    Pretty-print a PyTorch module with:
      - 4-space aligned indentation
      - parameter count next to each layer/module
      - total parameter count after each closing parenthesis
    """
    import torch.nn as nn

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    def layer_repr(m):
        # PyTorch's one-line repr for leaf modules
        return m.extra_repr() if hasattr(m, "extra_repr") else ""

    def format_module(m, name=None, level=0):
        indent = " " * (indent_size * level)
        child_indent = " " * (indent_size * (level + 1))

        total_params = count_params(m)
        children = list(m.named_children())

        # Header
        if name is None:
            lines = [f"{indent}{m.__class__.__name__}("]
        else:
            lines = [f"{indent}({name}): {m.__class__.__name__}("]

        # Leaf module
        if len(children) == 0:
            extra = layer_repr(m)
            if name is None:
                return [f"{indent}{m.__class__.__name__}({extra})  # params={total_params:,}"]
            return [
                f"{indent}({name}): {m.__class__.__name__}({extra})  # params={total_params:,}"
            ]

        # Recursive children
        for child_name, child in children:
            child_children = list(child.named_children())

            if len(child_children) == 0:
                extra = layer_repr(child)
                child_params = count_params(child)
                lines.append(
                    f"{child_indent}({child_name}): "
                    f"{child.__class__.__name__}({extra})  # params={child_params:,}"
                )
            else:
                lines.extend(format_module(child, child_name, level + 1))

        # Closing line with total params
        lines.append(f"{indent})  # total_params={total_params:,}")
        return lines

    print("\n".join(format_module(module)))

def print_class_attributes(obj, max_width=80):
    """
    Pretty-print all non-callable attributes of a class instance.

    Special handling:
      - torch.optim.Optimizer → print class name + (lr, weight_decay)
        with aligned indentation
    """
    import torch

    indent = " " * 4  # one "tab" (4 spaces)

    print("\n" + "=" * max_width)
    print(f"{obj.__class__.__name__} Attributes".center(max_width))
    print("=" * max_width)

    for key, value in sorted(vars(obj).items()):
        if callable(value):
            continue

        key_str = f"{key:>35} : "

        # -------------------------
        # Optimizer special case
        # -------------------------
        if isinstance(value, torch.optim.Optimizer):
            opt_name = value.__class__.__name__

            pg = value.param_groups[0]
            lr = pg.get("lr", None)
            wd = pg.get("weight_decay", None)

            # First line
            print(f"{key_str}{opt_name} (")

            # Indented fields
            print(f"{indent}{indent}{'lr':<15}: {lr}")
            print(f"{indent}{indent}{'weight_decay':<15}: {wd}")

            # Closing aligned under key
            print(f"{' ' * len(key_str)})")

        # -------------------------
        # Tensor handling
        # -------------------------
        elif hasattr(value, "shape"):
            val_str = f"{type(value).__name__}(shape={tuple(value.shape)})"
            print(f"{key_str}{val_str}")

        # -------------------------
        # NN modules
        # -------------------------
        elif isinstance(value, torch.nn.Module):
            print(f"{key_str}{value.__class__.__name__}(...)")

        # -------------------------
        # Default
        # -------------------------
        else:
            print(f"{key_str}{value}")

    print("=" * max_width + "\n")