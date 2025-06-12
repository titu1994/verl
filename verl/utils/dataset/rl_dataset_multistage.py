# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from omegaconf import ListConfig
import os
from typing import List, Union, Optional, Dict, Any
import copy
import pandas as pd
import json
from collections import defaultdict

import torch
import string
import yaml
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from verl.utils.dataset.rl_dataset import process_image, RLHFDataset

class RLHFDatasetMultistage(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self, *args, **kwargs):
        for stage_name, stage_config in self.config.get('stages', []).items():
            config_file = stage_config.prompt_config
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            self.prompt_configs[stage_name] = config
        assert 'start' in self.prompt_configs, "Start stage is required"
        super().__init__(*args, **kwargs)

    def apply_prompt_config(self, stage_name, raw_chat):
        prompt_config = self.prompt_configs[stage_name]
        for chat_item in raw_chat:
            role = chat_item['role']
            assert role in prompt_config
            content = chat_item['content'].format(**prompt_config[role])
            chat_item['content'] = content
        return raw_chat

    def build_prompt_features(self, row_dict, prompt_key=None):
        prompt_key = prompt_key or self.prompt_key
        chat = row_dict[prompt_key]

        if self.disable_chat_template:
            prompt_with_chat_template = chat
        else:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        raw_prompt = prompt_with_chat_template
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getitem__(self, item, stage_name='start'):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        return row_dict
        # row_dict = self.apply_prompt_config(row_dict, stage_name)
        # return self.build_prompt_features(row_dict)
