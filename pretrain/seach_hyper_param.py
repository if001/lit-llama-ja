import os
import redpajama


def main(
        out_dir:str = "", 
        log_dir: str = "",
        train_data_dir: str = "data/ja_full_data",
        val_data_dir: str = "data/ja_data"
        ):
    # batchs = [16, 32, 64, 128, 256]
    # batchs = [16, 64, 256]
    batchs = [4, 128]
    batchs = [4, 32, 128]
    # batchs = [4]
    # lrs = [1e-4, 5e-4, 1e-5, 5e-5, 1e-6]
    # lrs = [1e-3, 1e-4, 1e-5]
    lrs = [1e-4]
    models = ["phi-1_5-400M", "phi-1_5-400M_deep_layer", "phi-1_5-400M_multi_head"]
    models = ["phi-1_5-400M_deep_layer", "phi-1_5-400M_multi_head"]
    # weight_decays = [0.1, 0.01, 0.001, 0.0001]
    weight_decays = [0.001]

    skip_set = []

    ## for colab settings
    # out_dir="/content/drive/MyDrive/pre_trained/llama2/400M_search/phi-1_5/models"
    # out_dir="/content/drive/MyDrive/pre_trained/llama2/400M_search/phi-1_5/batchs"
    # log_dir="/content/drive/MyDrive/pre_trained/llama2/400M_search/phi-1_5/models/logs"
    # log_dir="/content/drive/MyDrive/pre_trained/llama2/400M_search/phi-1_5/batchs/logs"

    out_dir="./400M_search/phi-1_5/models"
    log_dir="./400M_search/phi-1_5/logs"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cnt = 0
    for batch in batchs:
        for lr in lrs:
            for weight_decay in weight_decays:
                for model in models:
                    print(f'start batch:{batch} lr:{lr} weight_decay:{weight_decay}')
                    if [batch, lr, weight_decay] in skip_set:
                        print('skip...', batch, lr, weight_decay)
                        continue
                    redpajama.main(
                        devices=1,
                        train_data_dir=train_data_dir,
                        val_data_dir=val_data_dir,
                        model_size=model,
                        out_dir=out_dir,
                        log_dir=log_dir,
                        batch_size=batch,
                        lr=lr,
                        weight_decay=weight_decay,
                        interrupt=True
                    )
                    print('-'*100)

if __name__ == "__main__":
    from jsonargparse.cli import CLI
    CLI(main)
