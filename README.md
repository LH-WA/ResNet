# NNDL-PJ2 — Train CNNs in CIFAR10

A project aimed at improving ResNet accuracy on CIFAR10 dataset while using less trainable parameters(under 5 million). The project is fully modular and the best accuracy reach **94.43%**. 

## **Documentation**

1. Project report can be found at ' docs/NNDL_PJ2.pdf '
2. Training process recorded in ' plots' 
3. Model weight saved in ' checkpoints '
4. Tensorboard recorded in ' logs '

## **Project Structure**

| Directory / File | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `checkpoints`    | Contains the saved network states                    |
| `docs`           | Project documentation                                |
| `logs`           | Log the best training process in tensorboard         |
| `plots`          | Saved plots                                          |
| `models.py`      | Resnet definitions and model configuration           |
| `resnet.py`      | main python file for training                        |
| `utils.py`       | Functions and Data Transform definitions             |
| `plot_tools.py`  | Draw Loss&Acc history and Confuse Matrix             |
| `Test.py`        | Load trained model for prediction (in test data set) |
| `Grad_cam.py`    | Visualize structure in each layers                   |

---
## **Run options**

To get help execute -
```bash
python resnet.py -h
```

**Optional Arguments:**

| Arguments | Variable | Description                  | Default|
|:-------------:|:------------:|:-------------------------:|:---------------:|
| -h      | --help        | Show this help message and exit||
| -dp \<path> | --data_path | Dataset storage path | CIFAR10/|
| -trb \<int>| --train_batch_size | Batch size for training | 128|
| -vab \<int> | --val_batch_size | Batch size for validation | 128 |
| -teb \<int> | --test_batch_size | Batch size for testing | 128 |
| -e \<int>  | --epochs | Number of epochs for training | 1|
|    -m \<str>    |           --model           |              Load ResNet Model from `models.py` <br> ResNet10, ResNet16, ResNet24, ResNet48              | ResNet24 |
|    -o \<str>    |         --optimizer         | Optimizer for Network <br>Choices: SGD , SGDN, Adagrad , Adadelta , |   SGD    |
|  -lr \<float>   |       --learning_rate       |                  Learning Rate of optimizer                  |   0.1    |
|   -lrm \<str>   |          --lr_mode          | Switch Learning Mode<br>Choices: Step, Exp, Cosine, Dynamic  |   Exp    |
|   -lp \<path>   |         --load_path         |               Path to Model weight and params                |   None   |
|  -mo \<float>   |         --momentum          |                    Momentum of optimizer                     |   0.9    |
|  -wd \<float>   |       --weight_decay        |                  Weight Decay of optimizer                   |   1e-3   |
|   -act \<str>   |        --activation         |          Activations: ReLU, ReLU6, LRU, ELU, Swish           |   ReLU   |
| -coe \<float> |       --activation_coe       |                    Activation coefficient                    |   0.1    |
| -mix | --mixup | Enable Mixup Argumentation | false |
|       -ns       |        --net_summary        |                    Print Network Summary                     |  false   |
|       -sm       |        --save_model         |                  Save Params of Best Model                   |  false   |
| -sw | --save_weight   | Save Weight of Best Model | false|
| -st | --save_tbd |           Save plots and net graph in Tensorboard            | false|
|      -shis      |         --save_his          |                Save Loss, Acc and Lr history                 |  false   |
|      -scm       |       --save_conf_mat       |                    Save confusion matrix                     |  false   |

**Usage**
> resnet.py [-h] [-dp <path>] [-trb <int>] [-vab <int>] [-teb <int>] [-e <int>] [-m <str>] [-o <str>] [-lr <float>] [-lrm <str>] [-lp <path>] [-mo <float>] [-wd <float>] [-act <str>] [-coe <float>] [-mix] [-ns] [-sm] [-sw] [-st] [-shis] [-scm]

---
## **How to run**

1. Use optimizer 'SGDN', learning scheduler 'Exp', activation 'ReLU6' train ResNet24 model in 100 epochs. 
    ```bash
    python resnet.py -e 100 -o SGDN -lrm Exp -act ReLU6 -m ResNet24
    ```

2. Run ResNet 24 and save history plot, confuse matrix, model params, model state_dict, tensorboard.
    ```bash
    python resnet.py -shis -scm -sm -sw -st
    ```
3. See the saved records in tensorboard.
    ```bash
    python -m tensorboard.main --logdir="<FilePath>\ResNet\logs"
    ```
    > Replace `<FilePath>` absolute path
