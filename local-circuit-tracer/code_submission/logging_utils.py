# logging_utils.py
import os
import sys
import time
import datetime
import logging
import numpy as np
from transformers.trainer_callback import TrainerCallback
import pynvml

logger = logging.getLogger(__name__)


def get_gpu_utilization():
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None).split(",")
    out_str = []
    for i in range(deviceCount):
        if str(i) not in cuda_visible:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        out_str.append(f"({i}): {info.used//1024**2}/{info.total//1024**2} MB")    
    return f"[[GPU] {', '.join(out_str)}]"

class LogFlushCallback(TrainerCallback):
    """ Like printer callback, but with logger and flushes the logs every call """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self.logger.info(logs)
        sys.stdout.flush()


def load_logger(logger):
    class LogFormatter():
        def __init__(self):
            self.start_time = time.time()

        def format(self, msg):
            time_passed = round(msg.created - self.start_time)
            gpu_usage = get_gpu_utilization()
            prefix = "[%s - %s]" % (
                time.strftime('%x %X'),
                datetime.timedelta(seconds=time_passed)
            )
            prefix = prefix + gpu_usage
            msg_text = msg.getMessage()
            return "%s %s" % (prefix, msg_text) if msg_text else ''

    logger.setLevel(logging.INFO)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(LogFormatter())
    logger.addHandler(log_handler)
    return logger

def write_output_to_file(eval_pred, tokenizer, output_path):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    preds = [] # detokenized outputs
    labs = [] # detokenized labels
    for b in range(predictions.shape[0]):
        
        l_eos_idx = np.argmax(labels[b]==tokenizer.eos_token_id)
        lab = labels[b][:l_eos_idx]
        p_eos_idx = np.argmax(predictions[b]==tokenizer.eos_token_id)
        if p_eos_idx == 0:
            p_eos_idx = len(predictions[b])
        pred = predictions[b][1:p_eos_idx]

        dt_lab = tokenizer.decode(lab)
        dt_pred = tokenizer.decode(pred)

        labs.append(dt_lab)
        preds.append(dt_pred)

    with open(output_path + "hyp.txt", 'w', encoding='utf8') as file:
        for line in preds:
            file.write(f"{line}\n")

    with open(output_path + "ref.txt", 'w', encoding='utf8') as file:
        for line in labs:
            file.write(f"{line}\n")


class SaveTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer, output_dir):
        super().__init__()
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(self.output_dir, "checkpoint-{}".format(state.global_step))
        if state.is_world_process_zero:
            self.tokenizer.save_pretrained(checkpoint_dir)