# Copyright (c) 2025 Alibaba PAI Team.
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

import numpy as np
import io
import copy
import json
import torch
try:
    from megatron import get_args
except:
    from megatron.training import get_args
from datasets import load_dataset
from tqdm import tqdm

from megatron_patch.tokenizer import get_tokenizer

class JSONRLHFDataset(torch.utils.data.Dataset):
    """
    Experimental: This dataset is aimed for SFT of arbitrary models with a default chat_template, 
    but not tested on all cases.

    A class for processing a conversation dataset
    """

    def __init__(self, path, max_padding_length, split='train'):
        super().__init__()
        self.tokenizer = get_tokenizer()
        assert hasattr(self.tokenizer, 'apply_chat_template'), \
            "The SFT-Raw Dataset is valid for tokenizers with chat template, please provide a template."
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.is_pad_token_eos_token = self.tokenizer.pad_token_id == self.eos_token_id
        self.max_seq_length = max_padding_length //2

        list_data_dict = load_dataset(
            'json',
            data_files=path[0],
            split=split,
        )

        train_dataset = list_data_dict.map(
            self.preprocess,
            batched=True,
            batch_size=3000,
            num_proc=16,
            remove_columns=list_data_dict.column_names,
            load_from_cache_file=False,
            # desc="Running Encoding"
        )

        self.input_ids = train_dataset['input_ids']
        self.labels = train_dataset['lengths']
        self.ground_truths = train_dataset['ground_truths']
        self.samples = []

        for inputs, labels, ground_truths in tqdm(zip(self.input_ids, self.labels, self.ground_truths)):
            self.samples.append([inputs, labels, ground_truths])

        print('  >> total number of samples: {}'.format(len(self.samples)))

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode, encoding='utf-8')
        return f

    def jload(self, f, mode='r'):
        """
        Load a .json file into a dictionary.
        Args:
            f: The file object or string representing the file path.
            mode: The mode in which to open the file (e.g., 'r', 'w', 'a').
        Returns:
            A dictionary containing the contents of the JSON file.
        """
        f = self._make_r_io_base(f, mode)
        jdict = [json.loads(x) for x in f.readlines()]
        f.close()
        return jdict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def preprocess(self, examples):
        """
        Preprocess the data by tokenizing.
        Args:
            sources (List[str]): a list of source strings
            targets (List[str]): a list of target strings
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the examples
        """


        
        input_ids = []
        lengths = []
        ground_truths = []
        all_ground_truths = [prompt for prompt in examples['ground_truth']]
        chats = [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
            ] for prompt in examples['query']]
        for i,chat in enumerate(chats):
            source = self.tokenizer.apply_chat_template(
            chat,
            # tokenize=True,
            # add_generation_prompt=True
            )
            if len(source) >= self.max_seq_length:
                continue
            input_ids.append(source)
            lengths.append(len(source))
            ground_truths.append(all_ground_truths[i])
        return dict(input_ids=input_ids, lengths=lengths, ground_truths=ground_truths)

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_id, length, ground_truth = sample
        input_id = torch.tensor(input_id, dtype=torch.int64)
        train_sample = {
            "text": input_id,
            "length": input_id.shape[0],
            "ground_truth": ground_truth,
            "loss_multiplier": True,        
        }

        return train_sample
