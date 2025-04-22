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
from typing import List, Union
import copy
import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import yaml
import json

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def format_prompt_config(prompt_config, row_dict):
    new_data = {}
    for key, value in prompt_config.items():
        if isinstance(value, dict):
            new_data[key] = format_prompt_config(value, row_dict)
        else:
            try:
                new_data[key] = value.format(**row_dict)
            except:
                new_data[key] = value
    return new_data

def apply_prompt_config(prompt_config, prompt_template, tokenizer):
    def apply_prompt_config(row_dict: dict):
        new_data = {}
        testtype = row_dict.get('testtype', None)
        new_data = format_prompt_config(prompt_config, row_dict)
        messages = []

        if 'prefix' in new_data:
            prompt = f"{tokenizer.bos_token}{new_data['prefix']} "
        else: 
            if 'system' in new_data:
                system = new_data['system']
                messages.append({'role': 'system', 'content': system})
            if 'user' in new_data:
                user = new_data['user']
                if (testtype is not None) and (new_data.get('user_addendum', {}).get(testtype, '') != ''):
                    user = f"{user}\n\n{new_data['user_addendum'][testtype]}"
                messages.append({'role': 'user', 'content': user})
            if 'assistant' in new_data:
                messages.append({'role': 'assistant', 'content': new_data['assistant']})

            prompt = f"{prompt_template['text_begin']}"
            for i, message in enumerate(messages):
                begin = prompt_template[f"{message['role']}_begin"]
                end = prompt_template[f"{message['role']}_end"]
                prompt += f"{begin}{message['content']}"
                if (i < len(messages) - 1) or (message['role'] != 'assistant'):
                    prompt += f"{end}\n"
            if messages[-1]['role'] != 'assistant':
                begin = prompt_template['assistant_begin']
                prompt += f"{begin}"
        return prompt
    return apply_prompt_config

def apply_tags(prompt_config, tag_name):
    def apply_tags(row_dict: dict):
        return prompt_config[tag_name]
    return apply_tags


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


def read_jsonl_as_dataframe(jsonl_file: str) -> pd.DataFrame:
    d = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            for k in line:
                if k not in d:
                    d[k] = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            for k in d:
                if k in line:
                    d[k].append(line[k])
                else:
                    d[k].append(None)
    return pd.DataFrame(d)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 disable_chat_template=True,
                 truncation='error',
                 prompt_config_files=None,
                 prompt_template_file=None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]
        if not isinstance(prompt_config_files, (List, ListConfig)):
            prompt_config_files = [prompt_config_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.disable_chat_template = disable_chat_template
        self.stop_phrases = []

        if prompt_config_files is not None:
            self.disable_chat_template = True # We already apply chat template using prompt template file
            assert prompt_template_file is not None, "Prompt template file is required when prompt config file is provided"
            assert os.path.exists(prompt_template_file), f"Prompt template file {prompt_template_file} does not exist"
            assert prompt_template_file.endswith('.yaml'), f"Prompt template file {prompt_template_file} must be a yaml file"
            self.prompt_configs = []
            for prompt_config_file in prompt_config_files:
                assert os.path.exists(prompt_config_file), f"Prompt config file {prompt_config_file} does not exist"
                assert prompt_config_file.endswith('.yaml'), f"Prompt config file {prompt_config_file} must be a yaml file"
                with open(prompt_config_file, 'r') as f:
                    self.prompt_configs.append(yaml.safe_load(f))
            with open(prompt_template_file, 'r') as f:
                self.prompt_template = yaml.safe_load(f)
            self.stop_phrases = [self.tokenizer.encode(phrase, add_special_tokens=False) for phrase in self.prompt_template['stop_phrases']]


        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            if parquet_file.endswith('.parquet'):
                dataframe = pd.read_parquet(parquet_file)
            elif parquet_file.endswith('.jsonl'):
                dataframe = pd.read_json(parquet_file, lines=True)
            else:
                raise ValueError(f'Unsupported file type: {parquet_file}')
            dataframes.append(dataframe)

        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        if self.prompt_configs is not None:
            for i in range(len(dataframes)):
                if len(self.prompt_configs) > 1:
                    assert len(self.prompt_configs) == len(dataframes), f"Number of prompt configs {len(self.prompt_configs)} must be equal to number of dataframes {len(dataframes)}"
                    dataframes[i][prompt_key] = dataframes[i].apply(apply_prompt_config(self.prompt_configs[i], self.prompt_template, tokenizer), axis=1)
                    dataframes[i]['think_tag'] = dataframes[i].apply(apply_tags(self.prompt_configs[i], 'think_tag'), axis=1)
                    dataframes[i]['answer_tag'] = dataframes[i].apply(apply_tags(self.prompt_configs[i], 'answer_tag'), axis=1)
                else:
                    dataframes[i][prompt_key] = dataframes[i].apply(apply_prompt_config(self.prompt_configs[0], self.prompt_template, tokenizer), axis=1)
                    dataframes[i]['think_tag'] = dataframes[i].apply(apply_tags(self.prompt_configs[0], 'think_tag'), axis=1)
                    dataframes[i]['answer_tag'] = dataframes[i].apply(apply_tags(self.prompt_configs[0], 'answer_tag'), axis=1)

        self.dataframe = pd.concat(dataframes)
        stop_phrases = self.stop_phrases + ([[self.tokenizer.eos_token_id]] if self.tokenizer.eos_token_id not in self.stop_phrases else [])
        stop_phrases = [stop_phrases for _ in range(len(self.dataframe))]
        self.dataframe['stop_phrases'] = stop_phrases
        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenize_fn = lambda doc: tokenizer.encode(doc[prompt_key]) if self.disable_chat_template else tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)
        if self.filter_prompts:
            self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                tokenize_fn(doc)) <= self.max_prompt_length, axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        if self.disable_chat_template:
            prompt_with_chat_template = chat
        else:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

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

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
