# EEE598HW1: MNIST Classification
**Homework01:** MNIST classification challenge with MLP and CNN 

Before running the experiments, clone the project: 

```bash
git clone https://github.com/mengjian0502/EEE598HW1_MNIST.git
```

**Important note:** Please change the `PYTHON` path inside all the `.sh` files before running the training or evaluation. 

## Training

To train the MLP model, please run:

```bash
bash run_mlp.sh
```

To train the CNN model, please run:

```bash
bash run_cnn.sh
```

## Evaluation

The pretrained MLP model located at: 

https://drive.google.com/drive/folders/10x9aVxM3blujBqvz3WQfl6DUir_4ttvi?usp=sharing

The pretrained CNN model located at:

https://drive.google.com/drive/folders/1j9bvmphn3sIzSb9wnFWjYMwHcAFRkHBW?usp=sharing

Given the pretrained model, please save the model to the `save` folder then change the . 

To evaluate the MLP model, please run

```bash
bash eval_mlp.sh
```

To evaluate the CNN model, please run

```bash
bash eval_cnn.sh
```



**Requirements:**

```bash
python 3.7.4
pytorch 1.7.1
```

