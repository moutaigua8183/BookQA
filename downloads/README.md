# Dataset #

The [OneDrive Link](https://1drv.ms/u/s!ArPzysVAJSvtpYMD8vBWfxGsZU3VqQ?e=BQxnak) contains our processed documents and train/val/test splits. Each document is cut into a list of 200-token trunks. The `dataset_splits.json` follows the format as below:
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

