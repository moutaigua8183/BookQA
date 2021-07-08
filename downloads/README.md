# Dataset #

Due the copyright, we are not allowed to directly share the datasets. Instead, we only release our data splits that follows the [NarrativeQA](https://github.com/deepmind/narrativeqa) setting but remove several probmatic samples. Please follow the steps [here](https://github.com/deepmind/narrativeqa) to download the full contents. The `dataset_splits.json` follows the format as below:
```json
{
    "train": [
        {
            "document_id":  "abcd1234",
            "type": "book"
        },
        {
            "document_id":  "9876wxyz",
            "type": "movie"
        },
        ...
    ],
    "val":  [
        { ... },
        { ... },
        ...
    ],
    "test":  [
        { ... },
        { ... },
        ...
    ]
}
```




# Models #

| Model Name           | <div style="width:150px">Description</div>         |  Link                |  
| :------------------- | :------------------------------------------------- | :------------------: |  
| Preread Reader       | A pretrained model with the text filling objective on NarrativeQA | [OneDrive](https://1drv.ms/u/s!ArPzysVAJSvtpO87Ls8wHsE_ndATJw?e=HjWQC5) | 
| Fine-tuned Reader    | A fine-tuned model on NarrativeQA                  | [OneDrive](https://1drv.ms/u/s!ArPzysVAJSvtpO9V1fZR6xo028Brjg?e=uDWH1T) | 

