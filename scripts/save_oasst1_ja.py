"""
oasst1の日本語翻訳されたデータセットをinstruction用のjsonファイルとして保存するscript

翻訳されたjsonファイルのダウンロード
!wget https://huggingface.co/datasets/kunishou/oasst1-89k-ja/resolve/main/oasst1_89k_ja.json


詳細は以下を参照
https://huggingface.co/datasets/kunishou/oasst1-89k-ja
"""

from datasets import load_dataset
import pandas as pd
import os
import json

from datasets import load_dataset
import pandas as pd
import os
import json

def prompt(instruct, output, input):
    if input:
        return f'以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{instruct}\n\n### 入力: {input}\n\n### 出力:\n{output}'
    return f'以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{instruct}\n\n### 出力:\n{output}'

def set_ppl(model_path, sp_model_path, data):
    import kenlm
    import sentencepiece
    import unicodedata
    print('process ppl...')

    model = kenlm.LanguageModel(model_path)
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(sp_model_path)

    def cal_ppl(text):
        text = unicodedata.normalize('NFD', text)
        tokens = sp.encode(text, out_type=str)
        sentence = " ".join(tokens)
        return model.perplexity(sentence)

    for v in data:
        if 'input' in v and v['input'] != "":
            v['input_ppl'] = cal_ppl(v['input'])
            _text = prompt(v['instruction'], v['output'], v['input'])
            v['full_ppl'] = cal_ppl(_text)
        else:
            v['input_ppl'] = 0
            _text = prompt(v['instruction'], v['output'], v['input'])
            v['full_ppl'] = cal_ppl(_text)
        v['instruction_ppl'] = cal_ppl(v['instruction'])
        v['output_ppl'] = cal_ppl(v['output'])
        
    return data

def main(
        json_file_path: str = "./oasst1_89k_ja.json",
        output_file_path: str = 'oasst1_ja_converted.json',
        model_path: str = "",
        sp_model_path: str = ""
    ):
    # oasst1のオリジナルデータのロード
    ds = load_dataset("OpenAssistant/oasst1")
    train = ds["train"].to_pandas()
    val = ds["validation"].to_pandas()

    df_origin = pd.concat([train, val], axis=0).reset_index(drop=True)

    # oasst1日本語翻訳データの読み込み
    df_ja = pd.read_json(json_file_path)

    # oasst1のオリジナルデータと日本語翻訳データのマージ
    df = pd.merge(df_origin, df_ja[["message_id", "text_ja"]], on="message_id", how="left").copy()
    df["text"] = df["text_ja"]

    df_assistant = df[(df.role == "assistant")].copy()
    df_prompter = df[(df.role == "prompter")].copy()
    df_prompter = df_prompter.set_index("message_id")
    df_assistant["output"] = df_assistant["text"].values

    inputs = []
    parent_ids = []
    for _, row in df_assistant.iterrows():
        input = df_prompter.loc[row.parent_id]
        inputs.append(input.text)
        parent_ids.append(input.parent_id)

    df_assistant["instruction"] = inputs
    df_assistant["parent_id"] = parent_ids

    df_assistant = df_assistant[
        ["instruction", "output", "message_id", "parent_id", "lang", "rank"]
    ].rename(columns={"message_id": "id"})


    # 翻訳タスクのみデータに異常があるので除外
    df_assistant2 = df_assistant[~df_assistant["instruction"].str.contains("翻訳")]

    learn_datas = []
    input_list = []

    for n in range(len(df_assistant2)):
        learn_data = {
            "instruction": str(df_assistant2.iloc[n, 0]),
            "input": "",
            "output": ""
        }

        input_list.append(df_assistant2.iloc[n, 0])
        learn_data["input"] = ""
        learn_data["output"] = str(df_assistant2.iloc[n, 1])

        learn_datas.append(learn_data)

    if model_path != '' and sp_model_path != '':
        learn_datas = set_ppl(model_path, sp_model_path, learn_datas)

    json_learn_data = json.dumps(learn_datas, indent=4, ensure_ascii=False)
    with open(output_file_path, 'w', encoding="utf-8") as f:
        f.write(json_learn_data)


if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)