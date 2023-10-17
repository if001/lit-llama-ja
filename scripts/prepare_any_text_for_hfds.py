"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from datasets import load_dataset, concatenate_datasets
from lit_llama.tokenizer import Tokenizer, HFTokenizer
from tqdm import tqdm


IGNORE_INDEX = -1


def prepare(
    save_path: Path = Path("data/any"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    test_split_ratio: float = 0.1,
    max_seq_length: int = 256,
    seed: int = 42,
    data_repo_id: str = "repo/id",
) -> None:
    """Prepare any dataset for finetuning (akin to Shakespheare full tuning).

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    def prepare(example):
        ins = example["instruction"]
        out = example["output"]
        if 'input' in example:
            inp = example['input']
            example["text"] = f'以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{ins}\n\n### 入力: {inp}\n\n### 出力:\n{out}'
        example["text"] = f'以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{ins}\n\### 出力:\n{out}'
        return example
    
    data_repo_id_list = []
    if "," in data_repo_id:
        data_repo_id_list = data_repo_id.split(",")
    else:
        data_repo_id_list = [data_repo_id]
    
    datasets = []
    for repo_id in data_repo_id_list:
        ds = load_dataset(repo_id, split="train")
        ds = ds.map(prepare).shuffle(seed=seed)
        print(repo_id)
        print(ds)
        print('-'*50)
        datasets.append(ds)
    ds = concatenate_datasets(datasets)
    print('merged')
    print(ds)
    dataset = ds.train_test_split(test_size=test_split_ratio)
    print('dataset[train]', dataset['train'])
    print('dataset[test]', dataset['test'])
    
    exit(0)
    print('load tokenizer...', tokenizer_path)    
    tokenizer = HFTokenizer(model_path=tokenizer_path)

    print("Processing train split ...")
    train_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(dataset['train'])
    ]
    torch.save(train_set, save_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(dataset['test'])
    ]
    torch.save(test_set, save_path / "test.pt")


def prepare_line(line: str, tokenizer: Tokenizer, max_length: int):
    """Processes a single sample.

    This function processes the line to produce the tokenized version of it.
    """    
    encoded_full_prompt = tokenize(tokenizer, line['text'], max_length=max_length, eos=True)
    return {
        "input_ids": encoded_full_prompt,
        "labels": encoded_full_prompt,
    }


def tokenize(
    tokenizer: Tokenizer, string: str, max_length: int, eos=True
) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
