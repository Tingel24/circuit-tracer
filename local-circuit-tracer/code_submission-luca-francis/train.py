# train.py
import argparse
import logging
import os
from functools import partial

import datasets

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    DataCollatorForLanguageModeling,
    DebertaV2ForMaskedLM,
    DebertaV2Tokenizer,
)
from transformers.trainer_callback import (
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
)
from transformers.trainer_seq2seq import Trainer
from transformers.training_args_seq2seq import TrainingArguments

from logging_utils import LogFlushCallback, SaveTokenizerCallback, load_logger
from model_utils import get_checkpoint_path, get_new_id, get_num_params

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = load_logger(logger)
datasets.logging.set_verbosity(logging.NOTSET) # patch to remove tqdm bars from dataloading

parser = argparse.ArgumentParser("Pretraining")
# Model args
parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="The model name." )
parser.add_argument("--models_folder", type=str, default="./models/", help="Folder to reload model from.")
parser.add_argument("--reload_id", type=str, default="", help="Job ID of model to reload.")
parser.add_argument("--new_id", type=str, default="", help="Job ID of new model. If not specified, a random one will be assigned.")
parser.add_argument("--reload_best", action="store_true", help="Reload best model. Done automatically for evaluation, \
                    but for training the default is the last.")
# Data args
parser.add_argument("--dataset", type=str, default="", help="Dataset used.")
parser.add_argument("--preprocessed", action="store_true", help="Whether to load preprocessed data.")
parser.add_argument("--spm_model", type=str, default="", help="Path of sentencepiece model. Only relevant for word models.")
parser.add_argument("--data_lim", type=int, default=0, help="Limit number of sentences to use, or 0 to disable.")
parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache data in. Set to /dev/shm/ for faster loading")
parser.add_argument("--max_seq_len", type=int, default=64, help="Maximum sequence length.")
parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for dataloading.")
# Training args
parser.add_argument("--debug", action="store_true", help="Activates debug mode, \
                    which shortens steps until evaluation, and enables tqdm.")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
parser.add_argument("--grad_acc", type=int, default=1, help="Accumulate gradients.")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
parser.add_argument("--warmup_steps", type=int, default=4000, help="Number of steps for warmup.")
parser.add_argument("--mask_prob", type=float, default=0.15, help="Probability of masking a token.")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluation.")
parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between saving.")
parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging.")
parser.add_argument("--log_to_file", action="store_true", help="Log to ./logs/jobid.log.")
parser.add_argument("--eval_only", action="store_true", help="Only evaluate.")
parser.add_argument("--bf16", action="store_true", help="bf16 training.")
parser.add_argument("--resume", action="store_true", help="Resume training, writing over old checkpoints.")
# Hub args
parser.add_argument("--push_to_hub", type=str, default="", help="Push to hub. Specify repo name.")
parser.add_argument("--push_only", action="store_true", help="Only push to hub.")
parser.add_argument("--register", type=str, default="", help="Register on AutoModel.")
parser.add_argument("--save_for_eval", type=str, default="", help="Save model for evaluation only.")
# Experimental args
parser.add_argument("--lora", action="store_true", help="Use LoRA. Only works if reloading a model")
parser.add_argument("--load_encdec", action="store_true", help="The model from reload_id is an encoder-decoder model")
parser.add_argument("--tiny", action="store_true", help="Use tiny model")
parser.add_argument("--small", action="store_true", help="Use small model")

class PrintOutputsCallback(TrainerCallback):
    """ Like printer callback, but with logger and flushes the logs every call """
    def __init__(self, logger, tokenizer):
        super().__init__()
        self.logger = logger
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ds = next(iter(kwargs["eval_dataloader"]))
        ds = {k: v[0].unsqueeze(0).to(model.device) for k, v in ds.items()}
        # print(ds["input_ids"].shape)
        out = model(**ds, return_dict=True)
        predictions = out.logits.argmax(dim=-1)
        labels = ds["labels"]
        source = ds["input_ids"]

        logger.info(f"source: {self.tokenizer.decode(source[0][:100])}")
        logger.info(f"label: {self.tokenizer.decode(labels.clamp(0)[0][:100])}")
        logger.info(f"pred: {self.tokenizer.decode(predictions[0][:100])}")

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def compute_acc(eval_pred):
    logits, labels = eval_pred

    predictions = logits#np.argmax(logits, axis=-1)
    predictions = predictions.flatten()
    labels = labels.flatten()
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]

    correct = labels == predictions
    accuracy = correct.sum() / float(len(correct))
    logger.info(f"Acc:{accuracy}")
    return {"acc": accuracy}


def tokenize_dataset(examples, tokenizer):
    encoded = {'input_ids':[]}#, 'attention_mask':[]}

    for i in range(len(examples['inputs'])):
        in_tok = tokenizer.encode(examples['inputs'][i]) # includes BOS and EOS
        
        encoded['input_ids'] += [in_tok]# + tgt_tok]

    return encoded


def tokenize_wiktionary(examples, tokenizer):
    encoded = {'input_ids':[]}#, 'attention_mask':[]}

    for i in range(len(examples['word'])):
        inp = []
        for key in examples:
            title = key.upper() + ":"
            ex = examples[key][i]
            if ex:
                inp.append(title)
                inp.append(ex)

        inp = " ".join(inp)
        in_tok = tokenizer.encode(inp) # includes BOS and EOS
        
        encoded['input_ids'] += [in_tok]# + tgt_tok]

    return encoded


def group_texts(examples, expanded_inputs_length):
    # Concatenate all texts.
    try:
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    except TypeError:
        print(examples)
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= expanded_inputs_length:
        total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
    # Split by chunks of max_len.

    result = {
        k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
        for k, t in concatenated_examples.items()
    }

    return result

if __name__ == "__main__":
    args = parser.parse_args()
    args.reload_best = args.reload_best or args.eval_only
    args.new_id = args.new_id if args.new_id else get_new_id(args)
    args.reload_id = args.reload_id if args.reload_id else args.new_id
    args.reload_path = f"{args.models_folder}{args.reload_id}"
    args.checkpoint_path = get_checkpoint_path(args.reload_path, args.reload_best)
    args.num_workers = os.cpu_count() if args.num_workers is None else args.num_workers

    if args.log_to_file:
        fh = logging.FileHandler(f'./logs/{args.new_id}.log')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    logger.info(args)
    torch.manual_seed(0)
    datasets.utils.logging.set_verbosity_error()


    if args.debug:
        args.logging_steps = 1
        args.eval_steps = 2


    args.tokenizer = DebertaV2Tokenizer(args.spm_model)
    config = AutoConfig.from_pretrained(args.model_name)
    config.vocab_size = args.tokenizer.vocab_size
    config.max_position_embeddings = 1024

    if args.tiny:
        config.hidden_size = 128
        config.intermediate_size = 512
        config.num_hidden_layers = 2
        config.num_attention_heads = 2

    if args.small:
        config.hidden_size = 256
        config.intermediate_size = 1024
        config.num_hidden_layers = 4
        config.num_attention_heads = 4

    if args.reload_id != args.new_id:
        model = DebertaV2ForMaskedLM.from_pretrained(args.checkpoint_path)
    else:
        model = DebertaV2ForMaskedLM(config)


    if args.lora:
        from peft import LoraConfig, get_peft_model
        from peft.utils.peft_types import TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules='all-linear',
        )
        model = get_peft_model(model, lora_config)


    if args.save_for_eval:
        model.save_pretrained(args.save_for_eval)
        args.tokenizer.save_pretrained(args.save_for_eval)
        exit()


    logger.info("----- Model Parameters -----")
    logger.info("Total: %d" % get_num_params(model))
    logger.info("----------------------------")

    dataset = DatasetDict()
    if args.preprocessed:
        dataset = load_from_disk(args.dataset)
    elif args.dataset == "wiktionary":
        # load csv
        dataset = load_dataset('csv', data_files='data/wiktionary_formatted.csv')
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=0)
        dataset['validation'] = dataset['test']
        del dataset['test']

    else:
        train_path, valid_path = args.dataset.split(",")
        dataset = load_dataset('text', data_files={'train': train_path, 'validation': valid_path})

        # reduce size of valid set
        dataset['validation'] = dataset['validation'].shuffle(seed=0)
        dataset['validation'] = Dataset.from_dict(dataset['validation'][:2000])

        dataset = dataset.rename_columns({'text': 'inputs'})


    if args.debug: # debug with smaller dataset
        if not args.eval_only:
            dataset['train'] = dataset['train'].filter(lambda example, idx: idx < 200, with_indices=True)
        dataset['validation'] = dataset['validation'].filter(lambda example, idx: idx < 200, with_indices=True)

    if args.data_lim: # limit amount of training data
        dataset['train'] = dataset['train'].filter(lambda example, idx: idx < args.data_lim, with_indices=True)

    tokenize = partial(tokenize_dataset, tokenizer=args.tokenizer)

    if args.dataset == "wiktionary":
        tokenize = partial(tokenize_wiktionary, tokenizer=args.tokenizer)

    group_texts = partial(group_texts, expanded_inputs_length=args.max_seq_len)

    if not args.eval_only:
        dataset['train'] = dataset['train'].map(tokenize, batched=True, num_proc=args.num_workers, remove_columns=dataset['train'].column_names)
        dataset['train'] = dataset['train'].map(group_texts, batched=True, num_proc=args.num_workers)

    dataset['validation'] = dataset['validation'].map(tokenize, batched=True, num_proc=args.num_workers, remove_columns=dataset['validation'].column_names)
    dataset['validation'] = dataset['validation'].map(group_texts, batched=True, num_proc=args.num_workers)

    print(dataset)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=args.tokenizer,
        mlm_probability=args.mask_prob
    )

    save_path = args.reload_path if args.resume else f"{args.models_folder}{args.new_id}"
    training_args = TrainingArguments(save_path,
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        log_level="info",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        bf16_full_eval=args.bf16,
        torch_compile=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        eval_accumulation_steps=1,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        load_best_model_at_end=False,
        resume_from_checkpoint=args.resume,
        disable_tqdm=not args.debug,)

    print_cb = LogFlushCallback(logger)
    print_ex = PrintOutputsCallback(logger, args.tokenizer)
    save_tokenizer_cb = SaveTokenizerCallback(args.tokenizer, save_path)

    total_steps = (len(dataset['train']) // args.batch_size) * args.epochs

    trainer = Trainer(
        model=model,
        tokenizer=args.tokenizer,
        args=training_args, 
        train_dataset=dataset['train'], 
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_acc,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[print_cb, print_ex, save_tokenizer_cb]
    )

    trainer.pop_callback(ProgressCallback if args.debug else PrinterCallback)

    # trainer.evaluation_loop = MethodType(evaluation_loop_new, trainer)

    if not args.eval_only and not args.push_only:
        trainer.train(resume_from_checkpoint=args.checkpoint_path if args.resume else None)
    
    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub, private=True)
        trainer.tokenizer.push_to_hub(args.push_to_hub, private=True)

    if args.push_only:
        exit()
    
    args.eval_only = True # for writing test set to file
    trainer.args.torch_compile = False # compile doesn't work with eval for some reason
    trainer.evaluate(dataset['validation'])