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

from .on_policy_runner import OnPolicyRunner
from .ts_runner import TSRunner
from .ee_runner import EERunner
from .cts_runner import CTSRunner
from .dreamwaq_runner import DreamWaQRunner
from .pact_runner import OnPolicyRunnerPACT
from .pact_pos_runner import OnPolicyRunnerPACTPos
from .postau_runner import OnPolicyRunnerPosTau
from .rl2ac_runner import OnPolicyRunnerRL2AC
from .abl1_runner import OnPolicyRunnerABL1
from .abl3_runner import OnPolicyRunnerABL3
from .kite_runner import OnPolicyRunnerKITE
from .unifp_runner import OnPolicyRunnerUniFP
from .b1_unifp_runner import OnPolicyRunnerB1UniFP
from .b1z1_pact_runner import B1Z1PACTRunner
from .b1z1_pact_pos_runner import B1Z1PACTPosRunner

from rsl_rl.utils.runner_registry import runner_registry
runner_registry.register("OnPolicyRunner", OnPolicyRunner)
runner_registry.register("TSRunner", TSRunner)
runner_registry.register("EERunner", EERunner)
runner_registry.register("CTSRunner", CTSRunner)
runner_registry.register("DreamWaQRunner", DreamWaQRunner)
runner_registry.register("PACTRunner", OnPolicyRunnerPACT)
runner_registry.register("PACTPosRunner", OnPolicyRunnerPACTPos)
runner_registry.register("PosTauRunner", OnPolicyRunnerPosTau)
runner_registry.register("RL2ACRunner", OnPolicyRunnerRL2AC)
runner_registry.register("ABL1Runner", OnPolicyRunnerABL1)
runner_registry.register("ABL3Runner", OnPolicyRunnerABL3)
runner_registry.register("KITERunner", OnPolicyRunnerKITE)
runner_registry.register("UniFPRunner", OnPolicyRunnerUniFP)
runner_registry.register("B1Z1PACTPosRunner", B1Z1PACTPosRunner)
runner_registry.register("B1UniFPRunner", OnPolicyRunnerB1UniFP)
runner_registry.register("B1Z1PACTRunner", B1Z1PACTRunner)
