from transformers import AutoTokenizer
import numpy as np
import torch
import json
import os
import sys
from pathlib import Path


def dissect(pretrained_model_name: str):
    """Explode pretrained weights into files

    Args:
        pretrained_model_name: One of
            * 'state-spaces/mamba-2.8b-slimpj'
            * 'state-spaces/mamba-2.8b'
            * 'state-spaces/mamba-1.4b'
            * 'state-spaces/mamba-790m'
            * 'state-spaces/mamba-370m'
            * 'state-spaces/mamba-130m'
                        
    Returns:
    """
    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file
    
    def load_state_dict_hf(model_name, device=None, dtype=None):
        resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                            _raise_exceptions_for_missing_entries=False)
        return torch.load(resolved_archive_file, weights_only=True)
    
    state_dict = load_state_dict_hf(pretrained_model_name)
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace('backbone.', '')
        new_state_dict[new_key] = state_dict[key]
    
    return new_state_dict


def main():
    if len(sys.argv) < 1:
        print("Please provide one of: mamba-130m, mamba-370m, mamba-790m, mamba-1.4b, mamba-2.8b.")
        return -1
    else:
        model_name = sys.argv[1] 
        print(f"Model to convert: {model_name}")

        state_dict = dissect(f'state-spaces/{model_name}')
        lens = []
        for key in state_dict:
            lens.append(len(state_dict[key].size()))
            foldername = Path(os.path.dirname(__file__)) / "converted" / model_name / key
            os.makedirs(foldername, exist_ok=True)
            data  = state_dict[key]
            dtype = 'f2' if any([x in key for x in ["embedding", "lm_head", "in_proj", "out_proj", "x_proj"]]) else 'f4'
            np_data = data.numpy().astype(dtype, order='C')
            metadata = {
                'key': key,
                'shape': data.shape,
                'is16bit': dtype == 'f2'
            }

            bytes = np_data.tobytes()
            with open(os.path.join(foldername, 'metadata.json'), 'w') as file:
                file.write(json.dumps(metadata))

            with open(os.path.join(foldername, 'weights.bin'), 'wb') as weightFile:
                weightFile.write(bytes)

if __name__ == "__main__":
    main()
