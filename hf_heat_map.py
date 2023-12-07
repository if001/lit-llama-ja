from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import seaborn as sns
import matplotlib.pyplot as plt

## matplotlibのlegendに日本語を使う
import japanize_matplotlib
japanize_matplotlib.japanize()

def main(
        prompt: str = "",
        model_name:str = "rinna/youri-7b",
        save_fig: Optional[str] = None,
        target_layer_idx: int = 0,
):    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokens = tokenizer.encode_plus(prompt, return_tensors='pt', add_special_tokens=True, max_length=512, truncation=True)

    outputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'], output_attentions=True)
    attentions = outputs.attentions[target_layer_idx].squeeze(0)

    figsize=(12, 12)
    plt.figure(figsize=figsize)

    for i, attention in enumerate(attentions):
        plt.subplot(4, 4, i+1)
        attention = attention.to(device='cpu', dtype=torch.float32).detach().numpy().copy()
        sns.heatmap(attention, cmap="YlGnBu", xticklabels=tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]), yticklabels=tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))
    
    if save_fig:
        plt.savefig(save_fig)
        print(f'save fig...{save_fig}')
    else:
        plt.show()

if __name__ =="__main__":
    from jsonargparse import CLI
    CLI(main)