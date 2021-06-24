# Prepare Data For Prereading (Pretraining) # 

The following script is to mask out 15% of the words with `<mask>` from the original content. The resulting data will be saved in a new folder located at the same level as the original folder. Note that the procedure will convert everything to lower case.
```bash
python convert_text_filling_data.py --data_dir=../sample_data/data
```
`--data_dir` is required. It is the folder that contains train/val/test source and target files.


# Train Models #

The backbone is BART model. Most of the training parameters can be left unchanged.

## Train a preread model ##
```bash
python train.py                                 \
    --data_dir=../sample_data/data_masked/      \
    --model_name_or_path=facebook/bart-large    \
    --tokenizer_name=facebook/bart-large        \
    --do_train                                  \
    --gpus=2                                    \
    --num_train_epochs=3                        \
    --learning_rate=5e-6                        \
    --train_batch_size=8                        \
    --max_target_length=900                     \
    --val_max_target_length=900                 \
    --test_max_target_length=900                \
    --cache_dir=pretrained                      \
    --output_dir=folder/where/preread/model/be/saved 
```
`--device` is to assign a specific GPU (cuda:{device}) to testing.  
`--cache_dir` is optional. It is to indicate the directory where the pretrained models and configurations are located.



## Finetune prepread reader ##
```bash
python train.py                                 \
    --data_dir=../sample_data/data_for_fid/     \
    --model_name_or_path=facebook/bart-large    \
    --tokenizer_name=facebook/bart-large        \
    --config_name=facebook/bart-large           \
    --ckpt_path=folder/where/preread/model/be/saved/epoch=2.ckpt  \
    --do_train                                  \
    --gpus=8                                    \
    --num_train_epochs=3                        \
    --learning_rate=5e-6                        \
    --train_batch_size=8                        \
    --max_target_length=900                     \
    --val_max_target_length=900                 \
    --test_max_target_length=900                \
    --cache_dir=pretrained                      \
    --output_dir=folder/where/finetuned/model/be/saved  
```

`--device` is to assign a specific GPU (cuda:{device}) to testing.  
`--cache_dir` is optional. It is to indicate the directory where the pretrained models and configurations are located.  
`--ckpt_path` is optional. It is the ckpt file of a specific preread model. To fine-tune a pretrained BART, simply ignore `ckpt_path`.  
The `output_dir` folder needs to be different from `ckpt_path`, since the numbering of checkpoint will be reset.


## Finetune huggingface pretrained LM ##
```bash
python train.py                                             \
    --data_dir=../sample_data/data_for_fid/                 \
    --model_name_or_path=facebook/bart-large                \
    --tokenizer_name=facebook/bart-large                    \
    --do_train                                              \
    --gpus=2                                                \
    --num_train_epochs=3                                    \
    --learning_rate=5e-6                                    \
    --train_batch_size=8                                    \
    --max_target_length=900                                 \
    --val_max_target_length=900                             \
    --test_max_target_length=900                            \
    --cache_dir=pretrained                                  \
    --output_dir=folder/where/finetuned/model/be/saved 
```
`--device` is to assign a specific GPU (cuda:{device}) to testing.  
`--cache_dir` is optional. It is to indicate the directory where the pretrained models and configurations are located.

## Continue Training a Checkpoint ##
```bash
python train.py               \
    --data_dir=path/to/data/folder                          \
    --model_name_or_path=facebook/bart-large                \
    --tokenizer_name=facebook/bart-large                    \
    --config_name=facebook/bart-large                       \
    --ckpt_path=folder/where/model/be/saved/epoch=2.ckpt    \
    --do_train                                              \
    --gpus=8                                                \
    --num_train_epochs=3                                    \
    --learning_rate=5e-6                                    \
    --train_batch_size=8                                    \
    --max_target_length=900                                 \
    --val_max_target_length=900                             \
    --test_max_target_length=900                            \
    --cache_dir=pretrained                                  \
    --output_dir=folder/where/new/model/be/saved  
```

`--device` is to assign a specific GPU (cuda:{device}) to testing.  
`--cache_dir` is optional. It is to indicate the directory where the pretrained models and configurations are located.  
`--ckpt_path` is the ckpt file.    
The `output_dir` folder needs to be different from `ckpt_path`, since the numbering of new models will be reset.



# Generate Result #

Most of the generation parameters can be left unchanged.

```bash
python generate.py                                              \
    --model_name_or_path=facebook/bart-large                    \
    --config_name=facebook/bart-large                           \
    --tokenizer_name=facebook/bart-large                        \
    --ckpt_path=folder/where/new/model/be/saved/epoch=2.ckpt    \
    --input_path=path/to/val.source                             \
    --task=summarization                                        \
    --bs=1                                                      \
    --device=1                                                  \
    --cache_dir=pretrained/                                     \
    --output_dir=folder/where/predictions/be/saved  
```  

`--device` is to assign a specific GPU (cuda:{device}) to testing.  
`--cache_dir` is optional. It is to indicate the directory where the pretrained models and configurations are located.  
    



# Evaluate Result #

```bash
cd ../evaluation
python  evaluate_test_results.py        \
    --target_file=path/to/val.target    \
    --result_file=path/to/val_hypo.txt 
```
Installation can be found in *evaluation* folder.

