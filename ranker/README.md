# Train Distantly Supervised Rankers #

The following scripts are to train a distantly supervised (DS) BERT ranker. Please refer to [data_for_ds_ranker](../sample_data/data_for_ds_ranker) for the training/testing data format.



## For Training With BERT Ranker #

```bash
python train_ds_ranker.py            \
    --data_dir=path/to/data/folder          \
    --model_type=bert                       \
    --model_name_or_path=bert-base-uncased  \
    --task_name=mnli                        \
    --do_train                              \
    --max_seq_length=256                    \
    --per_gpu_train_batch_size=32           \
    --learning_rate=3e-5                    \
    --num_train_epochs=2.0                  \
    --max_steps=32000                       \
    --cache_dir=pretrained                  \
    --output_dir=folder/where/ranker/model/be/saved  \
    --logging_steps=8000                    \
    --save_steps=4000
```
 
`--cache_dir` is optional. It is to indicate the directory where the pretrained models and configurations are located.



## For Ranking Passages With BERT Ranker ##

```bash
python rank_passage.py                          \
    --data_dir=path/to/data/folder              \
    --model_type=bert                           \
    --model_name_or_path=path/to/data/folder    \
    --task_name=mnli                            \
    --do_rank                                   \
    --rank_filename_without_ext=test            \
    --max_seq_length=256                        \
    --per_gpu_rank_batch_size=64                \
    --cache_dir=pretrained                      \
    --output_dir=folder/where/ranking/results/be/saved
```

`--cache_dir` is optional. It is to indicate the directory where the pretrained models and configurations are located.










