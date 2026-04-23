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
import os
import copy
import json
import hashlib
import torch
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from megatron import get_args
except:
    from megatron.training import get_args
from datasets import load_dataset
from tqdm import tqdm

from megatron_patch.tokenizer import get_tokenizer


def _tokenize_chunk(chunk, tokenizer_path, max_seq_length):
    """Worker function for parallel tokenization."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False, trust_remote_code=True
    )
    results = []
    for item in chunk:
        prompt = item["prompt"]
        chats = [{"role": "user", "content": prompt}]
        source = tokenizer.apply_chat_template(chats)
        if len(source) >= max_seq_length:
            continue
        results.append([source, len(source), item["label"]])
    return results


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
        self.max_seq_length = max_padding_length

        cache_path = self._get_cache_path(path[0], max_padding_length)
        if cache_path and os.path.exists(cache_path):
            print(f'  >> loading cached dataset from {cache_path}')
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
            print(f'  >> total number of samples: {len(self.samples)}')
            return

        jdict = self.jload(path[0])
        args = get_args()
        tokenizer_path = getattr(args, 'load', None)

        num_workers = min(16, max(1, os.cpu_count() // 2))
        if tokenizer_path and len(jdict) > 1000 and num_workers > 1:
            self.samples = self._parallel_tokenize(
                jdict, tokenizer_path, max_padding_length, num_workers
            )
        else:
            self.samples = self._serial_tokenize(jdict)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'  >> cached dataset to {cache_path}')

        print(f'  >> total number of samples: {len(self.samples)}')

    def _get_cache_path(self, data_path, max_padding_length):
        """Generate a deterministic cache path based on data file and config."""
        try:
            file_stat = os.stat(data_path)
            cache_key = f"{data_path}_{file_stat.st_size}_{file_stat.st_mtime}_{max_padding_length}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_dir = os.path.join(os.path.dirname(data_path), '.cache')
            return os.path.join(cache_dir, f"rlhf_{cache_hash}.pkl")
        except Exception:
            return None

    def _serial_tokenize(self, jdict):
        """Original serial tokenization path."""
        samples = []
        for item in tqdm(jdict, desc="Tokenizing"):
            prompt = item["prompt"]
            chats = [{"role": "user", "content": prompt}]
            source = self.tokenizer.apply_chat_template(chats)
            if len(source) >= self.max_seq_length:
                continue
            samples.append([source, len(source), item["label"]])
        return samples

    def _parallel_tokenize(self, jdict, tokenizer_path, max_padding_length, num_workers):
        """Parallel tokenization using multiprocessing."""
        chunk_size = max(1, len(jdict) // num_workers)
        chunks = [jdict[i:i + chunk_size] for i in range(0, len(jdict), chunk_size)]

        samples = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_tokenize_chunk, chunk, tokenizer_path, max_padding_length)
                for chunk in chunks
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel tokenizing"):
                samples.extend(future.result())
        return samples

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode, encoding='utf-8')
        return f

    def jload(self, f, mode='r'):
        """Load a .jsonl file into a list of dicts."""
        f = self._make_r_io_base(f, mode)
        jdict = []
        for line in f:
            line = line.strip()
            if line:
                jdict.append(json.loads(line))
                if len(jdict) >= 100000:
                    break
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
        all_ground_truths = [prompt for prompt in examples['answer']]
        chats = [[
        {"role": "system", "content": "Your are a helpful assistant."},
        {"role": "user", "content": prompt}] for prompt in examples['prompt']]
        for i,chat in enumerate(chats):
            source = self.tokenizer.apply_chat_template(
            chat,
            )
            if len(source) >= self.max_seq_length:
                print(len(source))
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
