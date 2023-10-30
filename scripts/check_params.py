import glob
import json

def main(
        target_dir: str = "./",

):    
    d = f"{target_dir}/search_param-*.json"
    files = glob.glob(d)
    for file in files:        
        with open(file, 'r') as f:
            json_load = json.load(f)
            print(json_load)
            print('-'*100)

if __name__ == "__main__":
    from jsonargparse.cli import CLI
    CLI(main)
