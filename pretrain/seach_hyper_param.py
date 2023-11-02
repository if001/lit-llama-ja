import subprocess
import redpajama


base_cmd=[
    "pretrain/redpajama.py",
    "--devices 1"
    "--train_data_dir data/ja_full_data",
    "--val_data_dir data/ja_data",
    "--model_size \"Llama-2-400M-hf\"",
    "--out_dir \"/content/drive/MyDrive/pre_trained/llama2/400M_search\"",
    "--log_dir \"/content/drive/MyDrive/pre_trained/llama2/400M_search/logs_lr\"",
    "--interrupt" "true"
]

# batchs = [16, 32, 64, 128, 256]
# batchs = [16, 64, 256]
# batchs = [2, 4, 8]
batchs = [4]
# lrs = [1e-4, 5e-4, 1e-5, 5e-5, 1e-6]
lrs = [1e-3, 1e-4, 1e-5]

# weight_decays = [0.1, 0.01, 0.001, 0.0001]
weight_decays = [0.001]

skip_set = []

cnt = 0
for batch in batchs:
    for lr in lrs:
        for weight_decay in weight_decays:
            print(f'start batch:{batch} lr:{lr} weight_decay:{weight_decay}')
            if [batch, lr, weight_decay] in skip_set:
                print('skip...', batch, lr, weight_decay)
                continue
            redpajama.main(
                devices=1,
                train_data_dir="data/ja_full_data",
                val_data_dir="data/ja_data",
                model_size="Llama-2-400M-hf",
                out_dir="/content/drive/MyDrive/pre_trained/llama2/400M_search",
                log_dir="/content/drive/MyDrive/pre_trained/llama2/400M_search/logs",
                batch_size=batch,
                lr=lr,
                weight_decay=weight_decay,
                interrupt=True
            )
            print('-'*100)
