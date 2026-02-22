from pathlib import Path
from collections import Counter, defaultdict
import os, sys

from pathlib import Path
from typing import Tuple, Union

DATASET_FILE = "/workspace/layered/dataset-v2_1.toml"
SRC_FOLDER = '/workspace/layered/image_data_v2/'
MUSUBI_TUNER_PATH = "/workspace/layered/musubi-tuner"
MODEL_CHECKPOINT_PATH = "/workspace/layered/model/default"


def read_toml(toml_path: Union[str, Path]) -> Tuple[str, str]:
    """
    Read a TOML file and return (image_directory, cache_directory) from the first [[datasets]] entry.

    Requires:
      - Python 3.11+: uses built-in tomllib
      - Python <=3.10: install `tomli` (pip install tomli)
    """
    toml_path = Path(toml_path)

    # TOML parser (py>=3.11 has tomllib)
    try:
        import tomllib  # type: ignore
        loads = tomllib.loads
    except ModuleNotFoundError:
        import tomli  # type: ignore
        loads = tomli.loads

    data = loads(toml_path.read_text(encoding="utf-8"))

    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise KeyError("TOML must contain a non-empty [[datasets]] array")

    first = datasets[0]
    if not isinstance(first, dict):
        raise TypeError("First [[datasets]] entry is not a table/dict")

    image_dir = first.get("image_directory")
    cache_dir = first.get("cache_directory")

    if not isinstance(image_dir, str) or not image_dir.strip():
        raise KeyError("Missing or invalid 'image_directory' in first [[datasets]] entry")
    if not isinstance(cache_dir, str) or not cache_dir.strip():
        raise KeyError("Missing or invalid 'cache_directory' in first [[datasets]] entry")

    return image_dir, cache_dir

DEST_FOLDER, STORAGE_CACHE = read_toml(DATASET_FILE)
print('='*10)
print("Parameters:")
print(f"DATASET_FILE: {DATASET_FILE}")
print(f"SRC_FOLDER: {SRC_FOLDER}")
print(f"MUSUBI_TUNER_PATH: {MUSUBI_TUNER_PATH}")
print(f"MODEL_CHECKPOINT_PATH: {MODEL_CHECKPOINT_PATH}")
print('='*10)
print(f"From dataset file: \n\tImage folder is:{DEST_FOLDER}\n\tCache_storage folder is:{STORAGE_CACHE}")
print(f"Images source: {SRC_FOLDER}")
print("Agree to proceed (y/n)?")
answer = input()
if answer.lower().strip() != 'y':
    sys.exit(-1)

if not SRC_FOLDER.endswith('/'):
    SRC_FOLDER += '/'
if not DEST_FOLDER.endswith('/'):
    DEST_FOLDER += '/'
if not STORAGE_CACHE.endswith('/'):
    STORAGE_CACHE += '/'

def get_list_files(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

os.makedirs(STORAGE_CACHE + 'tmp', exist_ok=True)
if len(get_list_files(STORAGE_CACHE)) != 0:
    raise Exception(f"{STORAGE_CACHE} folder is not empty, please resolve before proceeding!!")
if len(get_list_files(STORAGE_CACHE + 'tmp')) != 0:
    raise Exception(f"{STORAGE_CACHE}tmp folder is not empty, please resolve before proceeding!!")
if len(get_list_files(DEST_FOLDER)) != 0:
    raise Exception(f"{DEST_FOLDER} folder is not empty, please resolve before proceeding!!")

pngs = []
texts = []
for filename in get_list_files(SRC_FOLDER):
    if filename.endswith('.txt'):
        texts.append(filename)
    elif filename.endswith('.png'):
        pngs.append(filename.split('.')[0].split('_')[0])

print('='*10)
print(f" Found Text: {len(texts)} and Pngs: {len(pngs)}")
print('='*10)

if len(texts) == 0 or (len(texts) != 1 and len(texts) != len(set(pngs))):
    raise Exception('Number of texts not 1 but also not equal to images!')

#Count file prefixes by number of elements
by_size = defaultdict(lambda : [])
for name, cnt in Counter(pngs).items():
    print(f"{name} has {cnt}")
    by_size[cnt].append(name)

single_text = len(texts) == 1
for batch_size in by_size:
    print(f'Working on batch_size: {batch_size}')
    for file_prefix in by_size[batch_size]:
        print(f'-->Copy {SRC_FOLDER}{file_prefix}.* and {SRC_FOLDER}{file_prefix}_* to {DEST_FOLDER}')
        os.system(f"cp {SRC_FOLDER}{file_prefix}.* {DEST_FOLDER}")
        os.system(f"cp {SRC_FOLDER}{file_prefix}_* {DEST_FOLDER}")
        if single_text:
            print(f'-->Copy {SRC_FOLDER}{texts[0]} to {DEST_FOLDER}{file_prefix}.txt')
            os.system(f"cp {SRC_FOLDER}{texts[0]} {DEST_FOLDER}{file_prefix}.txt")

    # need to execute
    print('-->Execute generate ... ')
    os.system(f"python {MUSUBI_TUNER_PATH}/src/musubi_tuner/qwen_image_cache_latents.py" +
    f" --dataset_config {DATASET_FILE}" +
    f" --vae {MODEL_CHECKPOINT_PATH}/qwen_image_layered_vae.safetensors" +
    " --model_version layered")
    
    
    os.system(f"python {MUSUBI_TUNER_PATH}/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py" +
    f" --dataset_config {DATASET_FILE}" +
    f" --text_encoder {MODEL_CHECKPOINT_PATH}/qwen_2.5_vl_7b_bf16.safetensors" +
    " --model_version layered")
    print('-'*10)

    print('-->Move tensors to tmp ... ')
    print(f'Move {STORAGE_CACHE}*.safetensors to {STORAGE_CACHE}tmp')
    os.system(f"mv {STORAGE_CACHE}*.safetensors {STORAGE_CACHE}tmp")
    print('-'*10)

    print('-->Clear image_data ... ')
    print(f'rm {DEST_FOLDER}*')
    os.system(f"rm {DEST_FOLDER}*")
    print('-'*10)


print('Move from tmp to cache folder... ')
print(f'Move {STORAGE_CACHE}tmp/* to {STORAGE_CACHE}')
os.system(f"mv {STORAGE_CACHE}tmp/* {STORAGE_CACHE}")
print('-'*10)
print('Done!')

