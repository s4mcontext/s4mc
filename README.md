# Semi-Supervised Semantic Segmentation via Marginal Contextual Information

![](https://github.com/s4mcontext/s4mc/blob/main/imgs/method.png?raw=true)

## Description:information_source:
This is an official PyTorch implementation of the "Semi-Supervised Semantic Segmentation via Marginal Contextual Information" paper submission for NeurIPS 2023.

The method utilize contextual information to produce higher quality and higher quantity of pseudo-labels.


  * [Description](#description-information_source)
  * [Results](#results-bar_chart)
  * [Installation](#Installation-writing_hand)
  * [Preperation](#Preperation-card_index_dividers)
  * [Training](#training-weight_lifting)
  * [License](#License-paperclip:)
  * [Acknowledgement](#Acknowledgement-copyright)
  
  
  ## Results :bar_chart:
Results for PASCAL VOC 12, using additional coarse annotated:

|Method |  1/16  | 1/8 | 1/4 | 1/2 | full | 
|:--- |:---:|:---:|:---:|:---:|:---:|
|Sup only|45.77 |54.92 |65.88 |71.69| 72.50|
|baseline |52.16| 63.47| 69.46| 73.73| 76.54|
|S4MC     |70.96| 71.6*| 75.41| 77.73| 80.58|
|S4MC  $\psi$   |**74.32**| **75.62**| **77.84**| **79.72**| **81.51**|


For all the results, please refer to the paper experiment section

A visual example for the results:

![](https://github.com/s4mcontext/s4mc/blob/main/imgs/res.png?raw=true)

## Installation :writing_hand:

> 
git clone https://github.com/s4mcontext/s4mc.git && cd s4mc
conda create -n s4mc
conda activate s4mc
pip install -r requirements.txt

You also need to download a backbone trained on ImageNet-1k by either:
*  use pretrained pythorch flag and save the path.
* download from [Google drive link].(https://drive.google.com/file/d/1nzSX8bX3zoRREn6WnoEeAPbKYPPOa-3Y/view?usp=sharing "Google drive link") (credit below)


## Preperation :card_index_dividers:

Before training the models please put the datasets in the `data` sub-directory.

For PASCAL VOC 2012:
follow [this instuction](https://github.com/zhixuanli/segmentation-paper-reading-notes/blob/master/others/Summary%20of%20the%20semantic%20segmentation%20datasets.md "this instuction") and download `PASCAL VOC 2012 augmented with SBD dataset.`

For Cityscapes:
Download "leftImg8bit_trainvaltest.zip" and "gtFine_trainvaltest.zip" from: https://www.cityscapes-dataset.com/downloads/

unzip all into data with the following structure:

    data
    ├── cityscapes
    │   ├── gtFine
    │   └── leftImg8bit
    ├── splits
    │   ├── cityscapes
    │   └── pascal
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        ├── JPEGImages
        ├── SegmentationClass
        ├── SegmentationClassAug
        └── SegmentationObject


4. Go over the [dependencies](#dependencies-floppy_disk).

## Training :weight_lifting:

For training a semi-supervised model you need to first set a config.
We've provided an example config for PASCAL. all the splits of data are provided in the data sub-directory as well, so simply change the config with the desired parameters and dataset.

To run the code distributed, go to experiments and run: 
```
python -m torch.distributed.launch --nproc_per_node=<#GPUs> --nnodes=1 ../train_semi.py --config=<path_to_config> --seed <random_seed> --name <exp_name>
```

where
`<#GPUs>` is the number of cuda devices avalible for distributed training.
`<path_to_config>` is the path to your config
`<random_seed>` to set a random seed for reproducability
`<exp_name>` will save the model and tensorboard with the experiment name

## License :paperclip:

This project is released under the [Apache 2.0 ](https://github.com/Haochen-Wang409/U2PL/blob/main/LICENSE "Apache 2.0 ") license.



## Acknowledgement :copyright:

This repository code is heavily based on [U2PL](https://github.com/Haochen-Wang409/U2PL) as well as the link for the pre-trained model provided in here.

