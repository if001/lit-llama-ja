from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk 
from transformers import AutoTokenizer

from trl.trainer.utils import ConstantLengthDataset
from datasets.builder import DatasetGenerationError
from datasets.arrow_writer import SchemaInferenceError


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
    
def _prepare_packed_dataloader(        
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        num_of_sequences,
        chars_per_token,
        formatting_func=None,
        append_concat_token=True,
        add_special_tokens=True,
    ):
    if dataset_text_field is not None or formatting_func is not None:
        if tokenizer is None:
            raise ValueError("You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`.")

        constant_length_iterator = ConstantLengthDataset(
                tokenizer,
                dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_seq_length,
                infinite=False,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=append_concat_token,
                add_special_tokens=add_special_tokens,
        )

        def data_generator(constant_length_iterator):
            for i in constant_length_iterator:
                yield i

        try:
            packed_dataset = Dataset.from_generator(
                data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
            )
        except (DatasetGenerationError, SchemaInferenceError):
            raise ValueError(
                "Error occurred while packing the dataset. Make sure that your dataset has enough samples to at least yield one packed sequence."
            )
        return packed_dataset
    else:
        raise ValueError(
            "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
        )
            
def main(
        tokenizer_name: str = "",
        dataset_name: str = "",
        output_dir: str = "",
        max_seq_length: int = 2048,
        num_of_sequences: int =1024,
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

    dataset_text_field="text"
    chars_per_token=3.6
    remove_unused_columns=True

    train_data = _prepare_packed_dataloader(
        dataset["train"], 
        tokenizer,         
        dataset_text_field,
        max_seq_length,        
        num_of_sequences,
        chars_per_token,
        remove_unused_columns
        )
    test_data = _prepare_packed_dataloader(
        dataset["test"], 
        tokenizer,         
        dataset_text_field,
        max_seq_length,        
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
