# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU checks of state routing contracts introduced by uniform grouping."""
import torch
from easymagpie_vllm_omni.codec.cuda_graph import uniform_codec_metadata
from easymagpie_vllm_omni.codec.uniform_groups import CodecUniformGroups


def mixed_metadata():
    metadata = uniform_codec_metadata(3, 2, torch.device("cpu"))
    metadata.num_reqs = 5
    metadata.num_decodes = metadata.num_decode_tokens = 2
    metadata.num_prefills = 3
    metadata.num_prefill_tokens = 6
    metadata.state_indices_tensor_d = torch.tensor([[8], [3]], dtype=torch.int32)
    metadata.state_indices_tensor_p = torch.tensor([9, 1, 7], dtype=torch.int32)
    metadata.has_initial_states_p = torch.tensor([False, True, False])
    return metadata


def test_grouping_preserves_shuffled_pages_and_per_request_initial_flags():
    metadata = mixed_metadata()
    old_pages = metadata.state_indices_tensor_p.clone()
    old_initial = metadata.has_initial_states_p.clone()
    grouped = CodecUniformGroups._group_metadata(metadata, [4, 2, 3], 2, torch.device("cpu"))
    assert grouped.state_indices_tensor_p.tolist() == [7, 9, 1]
    assert grouped.has_initial_states_p.tolist() == [False, False, True]
    assert grouped.query_start_loc_p.tolist() == [0, 2, 4, 6]
    assert grouped.num_prefills == 3 and grouped.num_decodes == 0
    assert torch.equal(metadata.state_indices_tensor_p, old_pages)
    assert torch.equal(metadata.has_initial_states_p, old_initial)


def test_decode_conversion_retains_history_and_does_not_reset_live_pages():
    metadata = mixed_metadata()
    grouped = CodecUniformGroups._group_metadata(metadata, [1, 0], 1, torch.device("cpu"))
    assert grouped.state_indices_tensor_p.tolist() == [3, 8]
    assert grouped.has_initial_states_p.tolist() == [True, True]
    assert grouped.query_start_loc_p.tolist() == [0, 1, 2]
    assert grouped.num_prefill_tokens == 2
    assert metadata.num_decodes == 2
    assert metadata.state_indices_tensor_d.tolist() == [[8], [3]]


def test_group_compatibility_rejects_wrong_decode_frame_and_request_counts():
    metadata = mixed_metadata()
    assert CodecUniformGroups._compatible(metadata, [1, 1, 2, 2, 2])
    assert not CodecUniformGroups._compatible(metadata, [2, 1, 2, 2, 1])
    assert not CodecUniformGroups._compatible(metadata, [1, 1, 2, 4])
    assert not CodecUniformGroups._compatible(metadata, [1, 1, 2, 2, 3])
