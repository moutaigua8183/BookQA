import argparse
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List


import argparse
import glob
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import MBartTokenizer, T5ForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right


try:
    from .callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
    from .utils import (
        ROUGE_KEYS,
        LegacySeq2SeqDataset,
        Seq2SeqDataset,
        assert_all_frozen,
        calculate_bleu,
        calculate_rouge,
        flatten_list,
        freeze_params,
        get_git_info,
        label_smoothed_nll_loss,
        lmap,
        pickle_save,
        save_git_info,
        save_json,
        use_task_specific_params,
        parse_numeric_cl_kwargs,
    )
except ImportError:
    from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
    from utils import (
        ROUGE_KEYS,
        LegacySeq2SeqDataset,
        Seq2SeqDataset,
        assert_all_frozen,
        calculate_bleu,
        calculate_rouge,
        flatten_list,
        freeze_params,
        get_git_info,
        label_smoothed_nll_loss,
        lmap,
        pickle_save,
        save_git_info,
        save_json,
        use_task_specific_params,
        parse_numeric_cl_kwargs,
    )

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"






class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, self.mode)
#         self.target_lens = {
#             "train": self.hparams.max_target_length,
#             "val": self.hparams.val_max_target_length,
#             "test": self.hparams.test_max_target_length,
#         }
#         self.decoder_start_token_id = None  # default to config
#         if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
#             self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
#             self.model.config.decoder_start_token_id = self.decoder_start_token_id
#         self.test_max_target_length = self.hparams.test_max_target_length
   

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)


    def generate(self, input_ids, attention_mask, **generate_kwargs):
        # pad_token_id = self.tokenizer.pad_token_id
        # source_ids, source_mask, y = SummarizationDataset.trim_seq2seq_batch(batch, pad_token_id)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            max_length=20,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=3,
            early_stopping=True,
            use_cache=False,
            **generate_kwargs
        )
        return generated_ids



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries_or_translations(
    examples: List[str],
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    prefix='',
    args=None,
    **generate_kwargs,
) -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""

    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)

    if "summarization" in args.task:
        model: SummarizationModule = SummarizationModule(args)
    else:
        model: SummarizationModule = TranslationModule(args)
    model = model.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.to('cuda:{}'.format(args.device))
    
    
    
    print('#############################################')
    print("# model is loaded from", args.ckpt_path)
    print('# Results will be saved in', out_file)
    print('# tokenizer.all_special_tokens =', model.tokenizer.all_special_tokens)
    print('# tokenizer.all_special_ids =', model.tokenizer.all_special_ids)
    print('#############################################')
    generate_kwargs['fuse_num']         = args.fuse_num

    
    # update config with task specific params
    use_task_specific_params(model, task)
    for examples_chunk in tqdm(list(chunks(examples, batch_size))):
        # examples_chunk = [prefix + text[:400] for text in examples_chunk]
        examples_chunk = [prefix + text for text in examples_chunk[0].split('\t')][:args.fuse_num]
        # batch = model.tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to('cuda:{}'.format(device)).data
        batch: Dict[str, torch.Tensor] = model.tokenizer.prepare_seq2seq_batch(
                examples_chunk,   
                return_tensors="pt",
                add_prefix_space=True,
                truncation=True, 
                padding="longest",
        ).to('cuda:{}'.format(device)).data

        summaries = model.generate(
            input_ids=batch['input_ids'].unsqueeze(0), #[:,:,:350],
            attention_mask=batch['attention_mask'].unsqueeze(0), #[:,:,:350],
            **generate_kwargs,
        )
        dec = model.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()
    fout.close()
    
    
    
    


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default='', help="will be added to the begininng of src examples"
    )
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    ################################
    parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
    parser.add_argument(
            "--ckpt_path",
            default=None,
            type=str,
            help='path tooo stored model checkpoints',
        )
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_name_or_path", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--test_max_target_length", type=int)
    parser.add_argument("--eval_max_length", type=int)
    parser.add_argument("--range", type=str, default=None, help="the line range to be tested")
    parser.add_argument(
            "--fuse_num",
            default=100,
            type=int,
            help='num of passage vector to fuse in decoder',
        )
    ################################
    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    parsed = parse_numeric_cl_kwargs(rest)

    model_name = args.ckpt_path.split('/')[-2]
    epoch_indx = args.ckpt_path.split('.')[-2][-1]
    line_range = None if args.range is None else args.range.split('-')
    data_type  = args.input_path.split('/')[1]
    subfolder  = os.path.join('./results', model_name + '_epoch' + epoch_indx + '_on_' + data_type)
    if 'test' in args.input_path:
        result_file= os.path.join(subfolder, 'test_hypo.txt' if line_range is None else 'test_hypo_{}-{}.txt'.format(line_range[0], line_range[1]))
    elif 'val' in args.input_path:
        result_file= os.path.join(subfolder, 'val_hypo.txt' if line_range is None else 'val_hypo_{}-{}.txt'.format(line_range[0], line_range[1]))
    elif 'train' in args.input_path:
        result_file= os.path.join(subfolder, 'train_hypo.txt' if line_range is None else 'train_hypo_{}-{}.txt'.format(line_range[0], line_range[1]))
    else:
        import sys
        print('Wrong --input_path')
        sys.exc_info()

    if not os.path.exists(subfolder):
        os.makedirs(subfolder)


    examples = [" " + x.rstrip() if "t5" in args.model_name_or_path else x.rstrip() for x in open(args.input_path).readlines()[::2]]
    if line_range:
        examples = examples[int(line_range[0]) : int(line_range[1])]

    generate_summaries_or_translations(
        examples,
        result_file,
        args.model_name_or_path,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        prefix=args.prefix,
        args=args,
        **parsed,
    )
    if args.reference_path is None:
        return








if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    run_generate()
