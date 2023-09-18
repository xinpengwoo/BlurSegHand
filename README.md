# Installation

1. Set the environment. We recommend using [**Anaconda**](https://www.anaconda.com/products/distribution).

```
 conda create -n BSHN python=3.8
 conda activate BSHN
 pip install -r requirements.txt
 python src/setup.py
```

2. Modify the `torchgeometry` following [**this**](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527). Without modification, you will meet `RuntimeError: Subtraction, the - operator...`.

3. Download the MANO model files (```mano_v1_2.zip```) from [**MANO website**](https://mano.is.tue.mpg.de/). Unzip and put ```mano_v1_2/models``` into ```src/utils/MANO```. To be specific, ```MANO_RIGHT.pkl``` and ```MANO_LEFT.pkl``` should be located in ```src/utils/MANO/models/MANO_RIGHT.pkl``` and ```src/utils/MANO/models/MANO_LEFT.pkl```, respectively.

# Dataset

Download the following datasets, then locate them in ```datasets```. Instruction is available at [**here**](assets/docs/BlurHand.md).

+ [BlurHand](https://drive.google.com/drive/folders/178q3oUQrOIJMKi0KHoRoQmWRGM8JZnMi)
+ [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/)

For the generation of our BlurSegHand dataset, please run the below command.

```
CUDA_VISIBLE_DEVICES=0,1 python src/data/BlurSegMask_generation.py -opt options/train/BlurHandNet_BH.yml
CUDA_VISIBLE_DEVICES=0,1 python src/data/BlurSegMask_generation.py -opt options/test/BlurHandNet_BH.yml
```

Our BlurSegHand dataset shares the same structure of [BlurHand](https://github.com/xinpengwoo/BlurSegHand/blob/main/assets/docs/BlurHand.md).

# Training

For training our BlurSegHandNet on BlurSegHand, please run the below command.

```
CUDA_VISIBLE_DEVICES=0,1 python src/train.py -opt options/train/BlurHandNet_BH.yml
```

You can check detailed training configurations in `options/train/BlurHandNet_BH.yml`.

In default, we used 2 GPUs, but you can change the configurations by modulating `num_gpus` in `.yml` file.

The training states and logs will be saved in `experiments/BlurHandNet_BH`.


# Testing

If you trained your own BlurSegHandNet following, please run the below command to test your model.

```
CUDA_VISIBLE_DEVICES=0 python src/test.py -opt options/test/BlurHandNet_BH.yml
```

If you want to inference the model on one image input, please run the below command.

```
CUDA_VISIBLE_DEVICES=0 python src/test.py -opt options/test/inference.yml
```

## Acknowledgement

Our code implementaions are motivated from the below codes. We thank the authors for sharing the awesome repositories.

- [BlurHand_RELEASE](https://github.com/JaehaKim97/BlurHand_RELEASE)
- [digit-interacting](https://github.com/zc-alexfan/digit-interacting)


## Contact
If you have any question, please email `xinpengwoo@gmail.com`.