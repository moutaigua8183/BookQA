from pathlib import Path
from os import listdir
import numpy as np  
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import argparse





def text_infilling_masking(doc):
    tokens = doc.split()
    total_len = len(tokens)
    occupied_pos = np.array([], dtype=np.int64)
    span_list = list()
    while True:
        # sample start_pos following uniform 
        start_pos = np.random.randint(total_len)
        # mask_length following Poisson with lambda = 3
        span_len = np.random.poisson(lam=3)
        end_pos = start_pos + span_len
        # check if any token in the new span has been masked alreayd
        span = np.arange(start_pos, end_pos)
        valid = not any(np.isin(span, occupied_pos))
        if not valid:
            continue
        # add to span_list
        span_list.append((start_pos, end_pos))
        # remember the occupied pos
        occupied_pos = np.append(occupied_pos, span)
        # to mask 15% of total tokens
        if len(occupied_pos) / total_len > 0.15:
            break
    
    # Masking with [MASK]
    new = list()
    span_list = sorted(span_list, key = lambda x: x[0])
    frm = 0
    for span in span_list:
        new += tokens[frm : span[0]] + ['<mask>']
        frm = span[1]
    new += tokens[frm : ] + ['</s>']
    if new[-1] != '</s>':
        new.append('</s>')
    return ' '.join(new)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str)
    args = parser.parse_args()

    # random_para_folder = Path('data/decoder_book_movie_para_300_train_only')
    # infilling_text_folder = Path('data/pretrain_book_movie_para300_train_only_text_infill')

    assert len(args.data_dir) > 0
    assert Path(args.data_dir).exists()


    orig_path   = Path(args.data_dir)
    new_path    = orig_path.parent / Path(orig_path.name + '_masked')
    new_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in listdir(orig_path) if isfile(orig_path / Path(f)) and  (f.endswith(".source") or f.endswith('.target'))]

    for f in files:
        print('[*] Processing {}'.format(f))
        if f == 'train.source': 
            with open(orig_path/Path(f), 'r') as fp:
                raw_passages = fp.readlines()
            with open(new_path/Path(f), 'w+') as fp:
                for passage in tqdm(raw_passages):
                    masked_passage = text_infilling_masking(passage[:-1].lower())
                    print(masked_passage, file=fp)
            print('[*] Masked {} is created in {}'.format(f, new_path))
        else:
            with open(orig_path/Path(f), 'r') as fp:
                raw_passages = fp.readlines()
            with open(new_path/Path(f), 'w+') as fp:
                for passage in tqdm(raw_passages):
                    toks = passage[:-1].lower().split()
                    if toks[-1] != '</s>':
                        toks.append('</s>')
                    print(' '.join(toks), file=fp)
            print('[*] {} is created to {}'.format(f, new_path))



