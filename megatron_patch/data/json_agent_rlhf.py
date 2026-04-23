import io
import os
import json
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from tqdm import tqdm

try:
    from megatron import get_args
except ImportError:
    from megatron.training import get_args

from megatron_patch.tokenizer import get_tokenizer


def _tokenize_agent_chunk(chunk, tokenizer_path, max_seq_length):
    """Worker function for parallel tokenization of agent RLHF data."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False, trust_remote_code=True
    )
    results = []
    for item in chunk:
        prompt = item["prompt"]
        tools = item.get("tools", [])
        system_prompt = item.get("system_prompt", None)

        chats = []
        if system_prompt:
            chats.append({"role": "system", "content": system_prompt})
        chats.append({"role": "user", "content": prompt})

        source = tokenizer.apply_chat_template(chats)
        if len(source) >= max_seq_length:
            continue

        results.append({
            "input_ids": source,
            "length": len(source),
            "ground_truth": item["label"],
            "tools": tools,
            "system_prompt": system_prompt or "",
        })
    return results


class JSONAgentRLHFDataset(torch.utils.data.Dataset):
    """Dataset for agent RLHF training with optional tool definitions.

    Supports JSON format:
    {
        "prompt": "...",
        "label": "...",
        "tools": ["python_execute", ...],       # optional
        "system_prompt": "You are a helpful..."  # optional
    }

    If 'tools' is absent, behaves identically to JSONRLHFDataset (single-turn).
    """

    def __init__(self, path, max_padding_length, split='train'):
        super().__init__()
        self.tokenizer = get_tokenizer()
        assert hasattr(self.tokenizer, 'apply_chat_template'), \
            "Tokenizer must support apply_chat_template."
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        self.max_seq_length = max_padding_length

        cache_path = self._get_cache_path(path[0], max_padding_length)
        if cache_path and os.path.exists(cache_path):
            print(f'  >> loading cached agent RLHF dataset from {cache_path}')
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
            print(f'  >> total number of agent RLHF samples: {len(self.samples)}')
            return

        jdict = self._jload(path[0])
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
            print(f'  >> cached agent RLHF dataset to {cache_path}')

        print(f'  >> total number of agent RLHF samples: {len(self.samples)}')

    def _get_cache_path(self, data_path, max_padding_length):
        """Generate a deterministic cache path based on data file and config."""
        try:
            file_stat = os.stat(data_path)
            cache_key = f"{data_path}_{file_stat.st_size}_{file_stat.st_mtime}_{max_padding_length}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_dir = os.path.join(os.path.dirname(data_path), '.cache')
            return os.path.join(cache_dir, f"agent_rlhf_{cache_hash}.pkl")
        except Exception:
            return None

    def _serial_tokenize(self, jdict):
        """Original serial tokenization path."""
        samples = []
        for item in tqdm(jdict, desc="Tokenizing agent RLHF"):
            prompt = item["prompt"]
            tools = item.get("tools", [])
            system_prompt = item.get("system_prompt", None)

            chats = []
            if system_prompt:
                chats.append({"role": "system", "content": system_prompt})
            chats.append({"role": "user", "content": prompt})

            source = self.tokenizer.apply_chat_template(chats)
            if len(source) >= self.max_seq_length:
                continue

            samples.append({
                "input_ids": source,
                "length": len(source),
                "ground_truth": item["label"],
                "tools": tools,
                "system_prompt": system_prompt or "",
            })
        return samples

    def _parallel_tokenize(self, jdict, tokenizer_path, max_padding_length, num_workers):
        """Parallel tokenization using multiprocessing."""
        chunk_size = max(1, len(jdict) // num_workers)
        chunks = [jdict[i:i + chunk_size] for i in range(0, len(jdict), chunk_size)]

        samples = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_tokenize_agent_chunk, chunk, tokenizer_path, max_padding_length)
                for chunk in chunks
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel tokenizing agent RLHF"):
                samples.extend(future.result())
        return samples

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode, encoding='utf-8')
        return f

    def _jload(self, f, mode='r'):
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
        sample = self.samples[idx]
        input_id = torch.tensor(sample["input_ids"], dtype=torch.int64)
        return {
            "text": input_id,
            "length": input_id.shape[0],
            "ground_truth": sample["ground_truth"],
            "tools": sample["tools"],
            "system_prompt": sample["system_prompt"],
            "loss_multiplier": True,
        }
