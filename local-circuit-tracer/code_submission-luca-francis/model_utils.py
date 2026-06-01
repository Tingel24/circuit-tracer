# model_utils.py
import os
import numpy as np
import random

def get_new_id(models_folder):
    reload_id = os.environ.get('SLURM_JOB_ID')
    if reload_id is None: # otherwise assign a random id
        chars = '0123456789'
        while True:
            reload_id = ''.join(random.choice(chars) for _ in range(5))
            if not os.path.isdir(f"{models_folder}{reload_id}"):
                break
        print("Reload id assigned is:", reload_id)
    return reload_id

def get_checkpoint_path(reload_path, best=True):
    import glob
    checkpoint_paths = glob.glob(reload_path + "/checkpoint-*")
    if len(checkpoint_paths) == 0:
        return None
    # always save best and last, so best will always have the lowest number
    checkpoint_step_nums = [int(x.split("-")[-1]) for x in checkpoint_paths]
    if best:
        best_checkpoint = np.argmin(checkpoint_step_nums)
        best_checkpoint_path = checkpoint_paths[best_checkpoint]
        return best_checkpoint_path
    else: # get last checkpoint instead
        last_checkpoint = np.argmax(checkpoint_step_nums)
        last_checkpoint_path = checkpoint_paths[last_checkpoint]
        return last_checkpoint_path

def get_num_params(model):
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total

def get_num_unfrozen_params(model):
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
    return total

