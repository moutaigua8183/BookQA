# Book QA #

Codes and resutls for the TACL paper *[Narrative Question Answering with Cutting-Edge Open-Domain QATechniques: A Comprehensive Study](https://arxiv.org/pdf/2106.03826.pdf)*.

## Installation ##
The implementation is based on [huggingface transformer 3.2](https://github.com/huggingface/transformers) and our customizations are directly made to the original source code. We provide the exact version in *transformers* folder that corresponds to the results in the paper. The following scripts have only been tested on Ubuntu 18.04 with Python 3.6 & 3.7.

It is noteworthy that, based on our tests, our customization based on huggingface transformer 4.7 can provide better results. We will provide the code in the future.


```bash
conda create -n trans32 python=3.7
conda activate trans32
cd transformers
pip install -e ".[testing]"
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -r ./examples/requirements.txt
pip install pytorch-lightning==0.8.5
```

For evaluation, more details can be found in [evaluation/](evaluation/) folder



## Citation ##
If you find this repo useful, please consider citing our paper:
```bibtex
@article{mou2021narrative,
  title={Narrative Question Answering with Cutting-Edge Open-Domain QA Techniques: A Comprehensive Study},
  author={Mou, Xiangyang and Yang, Chenghao and Yu, Mo and Yao, Bingsheng and Guo, Xiaoxiao and Potdar, Saloni and Su, Hui},
  journal={arXiv preprint arXiv:2106.03826},
  year={2021}
}
```

