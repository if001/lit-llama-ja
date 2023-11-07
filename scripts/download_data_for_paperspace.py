import requests

import zipfile
import os
import requests
from pathlib import Path
from typing import List
from tqdm import tqdm


def download_file(url, headers, destination):
    """
    指定されたURLからファイルをダウンロードし、指定されたディレクトリに保存する
    """
    local_filename = destination / Path(url).name
    file_size = int(requests.head(url).headers["content-length"])

    print(f'donwload {url}')
    res = requests.get(url, headers=headers, stream=True)
    pbar = tqdm(total=file_size, unit="B", unit_scale=True)
    res.raise_for_status()
    with open(local_filename, 'wb') as f:
        for chunk in res.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
    return local_filename

def main(
        targets: List[str] = [],
        hf_token: str = "",
        download_dir: str = "",
        extraction_dir: str = "",
        zip_sub_dir: str = "",
        destination_dir: str = ""
):
    headers = {
        "Authorization": f"Bearer {hf_token}"
    }    
    base_url='https://huggingface.co/datasets/if001/llama_train_bin/resolve/main/'
    zip_urls = [ base_url+v for v in targets]


    download_directory = Path(download_dir)    
    extraction_directory = Path(extraction_dir)    
    destination_directory = Path(destination_dir)
    
    download_directory.mkdir(parents=True, exist_ok=True)
    extraction_directory.mkdir(parents=True, exist_ok=True)
    destination_directory.mkdir(parents=True, exist_ok=True)
    
    for url in zip_urls:
        zip_filename = download_directory / Path(url).name                
        if not zip_filename.exists():
            download_file(url, headers, download_directory)

        print('unzip...')
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extraction_directory)
        
        print('move...')
        full_data_path = extraction_directory / zip_sub_dir
        for json_file in full_data_path.glob('*.bin'):
            json_file.rename(destination_directory / json_file.name)
        
        # full_dataディレクトリを削除する
        print('clean...')
        for item in full_data_path.iterdir():
            if item.is_dir():
                os.rmdir(item)
            else:
                item.unlink()
        full_data_path.rmdir()

if __name__ == "__main__":
    from jsonargparse.cli import CLI
    CLI(main)
