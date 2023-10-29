import subprocess

base_cmd=[
    "pretrain/redpajama.py",
    "--devices 1"
    "--train_data_dir data/ja_full_data",
    "--val_data_dir data/ja_data",
    "--model_size \"Llama-2-400M-hf\"",
    "--out_dir \"/content/drive/MyDrive/pre_trained/llama2/400M_search\"",
    "--log_dir \"/content/drive/MyDrive/pre_trained/llama2/400M_search/logs\"",
    "--interrupt" "true"
]

batchs = [16, 32, 64, 128, 256]
lrs = [1e-4, 5e-4, 1e-5, 5e-5, 1e-6]
weight_decays = [0.1, 0.01, 0.001, 0.0001]

cnt = 0
for batch in batchs:
    for lr in lrs:
        for weight_decay in weight_decays:
            cnt += 1
            add = [
                f"--batch_size {batch}",
                f"--lr {lr}",
                f"--weight_decay {weight_decay}"
            ]
            cmd = base_cmd.copy() + add
            subprocess.run(cmd)