from datasets import load_dataset, concatenate_datasets, load_from_disk 
from trl.trainer.utils import _prepare_dataset
from transformers import AutoTokenizer

"""sample

!python finetune/prepare_sft_dataset.py \
--tokenizer_name="if001/sentencepiece_ja" \
--dataset_name="izumi-lab/llm-japanese-dataset,kunishou/databricks-dolly-15k-ja,if001/oasst1_ja_ppl" \
--seq_length=2048 \
--batch_size=32 \
--trust_remote_code=True \
--output_dir=""
"""

def format_instruction(ds):
    input = ds['input']
    instruction = ds['instruction']
    output = ds['output']
    if input is None:
        text = f"""以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい
### 指示:
{instruction}

### 応答:
{output}"""
    else:
        text = f"""以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。
### 指示:
{instruction}

### 入力:
{input}

### 応答:
{output}"""
        ds['text'] = text
        return ds
    
def main(
        tokenizer_name: str = "",
        dataset_name: str = "",
        output_dir: str = "",
        trust_remote_code: bool = True
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)

    if ',' in dataset_name:
        datasets = []
        dataset_names = dataset_name.split(",")
        for name in dataset_names:
            ds = load_dataset(name, split="train")        
            ds = ds.select(range(10))
            ds = ds.shuffle().map(format_instruction)
            unused_key = list(ds.features.keys())
            unused_key.remove('text')
            ds = ds.remove_columns(unused_key)        
            datasets.append(ds)
        dataset = concatenate_datasets(datasets)
    else:
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.map(format_instruction)
        
    dataset = dataset.shuffle().train_test_split(test_size=0.1)

    packing=True
    dataset_text_field="text"
    max_seq_length=2048
    formatting_func=None
    num_of_sequences=1024
    chars_per_token=3.6
    remove_unused_columns=True

    train_data = _prepare_dataset(
        dataset["train"], 
        tokenizer, 
        packing, 
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns
        )
    test_data = _prepare_dataset(
        dataset["text"], 
        tokenizer, 
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,        
        num_of_sequences,
        chars_per_token,
        remove_unused_columns
    )
    print('train_data', train_data)
    print('test_data', test_data)
    train_data.save_to_disk(output_dir)
    print('save to...', output_dir)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
