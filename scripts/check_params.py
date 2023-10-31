import glob
import json
import matplotlib.pyplot as plt


def main(
        target_dir: str = "./",

):    
    d = f"{target_dir}/search_param-*.json"
    files = glob.glob(d)
    labels = []
    loss_arr = []
    val_loss_arr = []
    for file in files:
        with open(file, 'r') as f:
            obj = json.load(f)
            loss_arr.append(float(obj['loss']))
            val_loss_arr.append(float(obj['val_loss']))
            iter = obj['iter']
            file_name = file.replace("search_param-", "")
            labels.append(f'{iter}-{file_name}')

    t = range(len(labels))
    fig = plt.figure(figsize=[25,15])
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.grid()    
    ax1.plot(t, loss_arr, label="loss")
    ax1.set_xticklabels(labels)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.grid()
    ax2.plot(t, val_loss_arr, label="val_loss")
    ax2.set_xticklabels(labels)

    _labels = ax1.get_yticklabels()
    plt.setp(_labels, rotation=45, fontsize=10)
    plt.show()

if __name__ == "__main__":
    from jsonargparse.cli import CLI
    CLI(main)
