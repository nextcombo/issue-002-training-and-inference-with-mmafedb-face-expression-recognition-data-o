# Issue 002: Training and Inference with MMAFEDB Face Expression Recognition Data on Ubuntu 22.04 System Using PaddleX and PaddleClas

## Training and Reasoning MMAFEDB Face Expression Recognition Dataset on PaddleX and PaddleClas Based on Ubuntu22.04 System (Detailed Tutorial)

### Super practical Paddle image classification training and inference tutorial, helping deep learning research!

```
Test environment:
Ubuntu 22.04.4 LTS
cuda_11.8
Tesla P4 graphics card
500G disk space
```

>

Preview of the final directory structure of project source code

![](01-b2b263f7.png)

1. Environmental Preparation (Important ⭐⭐⭐)

##### Creating a Virtual Environment

```
# Creating the environment
conda create -n paddleclas python=3.10
# Activation environment entry
conda activate paddleclas
```

##### Install PaddlePaddle

Reference: https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html

```
# Installation instructions
python -m pip install paddlepaddle-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
# Verify if PaddlePaddle is available on GPU
python -c "import paddle;print(paddle.utils.run_check());print(paddle.__version__)"
# Output the following information, then install is successful, paddlepaddle-gpu can be used.
'''
Running verify PaddlePaddle program ... 
I0821 11:21:45.437101  7032 pir_interpreter.cc:1524] New Executor is Running ...
W0821 11:21:45.437536  7032 gpu_resources.cc:114] Please NOTE: device: 0, GPU Compute Capability: 6.1, Driver API Version: 12.4, Runtime API Version: 11.8
I0821 11:21:45.686959  7032 pir_interpreter.cc:1547] pir interpreter is running by multi-thread mode ...
PaddlePaddle works well on 1 GPU.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
None
'''
```

##### Install PaddleX

Refer to the reference: https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html

```
pip install paddlex==3.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# Additional Dependencies
pip install pypdfium2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

##### Install PaddleClas Plugin

Refer to the reference: https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html

```
# Access the source code repository (for training purposes, does not conflict with the aforementioned inference library)
git clone -b release/3.1 https://github.com/PaddlePaddle/PaddleX.git 
cd PaddleX
# Source code compilation
pip install -e .
# Installing Plugins
paddlex --install PaddleClas
# Output the following information, then install is successful, paddlecls can be used.
'''
......
Successfully built paddleclas
Installing collected packages: paddleclas
Successfully installed paddleclas-2.6.0
All packages are installed.
'''
```

2. Data Preparation (Important ⭐⭐⭐)

##### 2.1 Data Download and Production

Official Kaggle data download

Reference: https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression

```
# Referenced Download Method
curl -L -o ./mma-facial-expression.zip\
  https://www.kaggle.com/api/v1/datasets/download/mahmoudima/mma-facial-expression
```

**Create paddle format data** (Placed in the PaddleX source code directory)

The directory structure is:

```
datasets/
└── mmafedb/
    ├── images/
    ├── label.txt
    ├── train.txt
    └── val.txt
```

The content of the `label.txt` file is as follows (**7 categories**):

```
0 angry
1 disgust
2 fear
3 happy
4 neutral
5 sad
6 surprise
```

The content of the `train.txt` file is as follows (**92,968 records**):

```
images/train_angry_10012Exp0fighting_people_350.jpg 0
images/train_angry_10018Exp0fighting_people_429.jpg 0
images/train_angry_10025Exp0fighting_people_5.jpg 0
images/train_angry_10037Exp0fighting_people_573.jpg 0
......
```

The content of the `val.txt` file is as follows (**17356 records**):

```
......
images/val_surprise_Surprise.32265.jpg 6
images/val_surprise_Surprise.32266.jpg 6
images/val_surprise_Surprise.32283.jpg 6
images/val_surprise_Surprise.32284.jpg 6
```

The `images/` directory contains all the images, and the statistics are as follows:

```
Total JPG image count: 110324
Post-duplication resolution statistics:
Resolution (Width x Height)  Occurrences
---------------------------
48×48           110324
There are a total of 1 different resolutions.
```

PS: test.txt is temporarily **not used**.

>

At the end of the article, reply with the corresponding **keyword** to obtain the prepared **paddle format** training dataset.

##### 2.2 Data Verification

**Operations in the PaddleX source code directory**

Create a new shell and execute the script `check_data.sh` with the following content:

```
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./datasets/mmafedb
```

Execute the verification script, output success information

```
# Command line execution
bash check_data.sh
# The following information is output, which indicates the verification is successful. The result content can be seen in the ./output directory.
'''
Connecting to https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/PingFang-SC-Regular.ttf ...
Downloading PingFang-SC-Regular.ttf ...
[==================================================] 100.00%
Check dataset passed !
'''
```

Data distribution histogram, `histogram.png`

![](02-2a778285.png)

3. Start training (important ⭐⭐⭐)

**Operations in the PaddleX source code directory**

Create a new shell and execute the script `script_run_train.sh` with the following content:

```
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./datasets/mmafedb \
    -o Global.device=gpu:0 \
    -o Train.num_classes=7 \
    -o Train.epochs_iters=300 \
    -o Train.batch_size=128 \
    -o Train.pretrain_weight_path=./pretrain_weight/PP-LCNet_x1_0_pretrained.pdparams \
    -o Train.log_interval=20 \
    -o Train.eval_interval=100 \
    -o Train.save_interval=100
```

###### Details of Training Startup Parameters

|

Parameter name

 ||

Data type

 ||

Description

 ||

Example value

 |

| `mode` ||

str

 ||

Specify the pattern: check_dataset/train/evaluate/export/predict

 || `train` |
| `dataset_dir` ||

str

 ||

Dataset path

 || `./datasets/mmafedb` |
| `device` ||

str

 ||

Specify the equipment to be used

 || `gpu:0` |
| `num_classes` ||

int

 ||

Number of categories in the dataset

 || `7` |
| `epochs_iters` ||

int

 ||

Number of repetitive learnings by the model on the training data

 || `300` |
| `batch_size` ||

int

 ||

Training batch size

 || `128` |
| `pretrain_weight_path` ||

str

 ||

Pre-trained weight path

 || `./pretrain_weight/PP-LCNet_x1_0_pretrained.pdparams` |
| `log_interval` ||

int

 ||

Training log print interval

|| `yaml file specifies the training log print interval` |
| `eval_interval` ||

int

 ||

Model evaluation interval

`yaml file specified model evaluation interval` |
| `save_interval` ||

int

 ||

Model save interval

`yaml file specified model save interval`

Execute the training script, and the results are stored in `./output directory`

```
# Command line execution
bash run_train.sh
```

After the training ends, proceed to the ./output directory for operations.

Visualization metrics output, using the visualdl command

```
# Command line execution
visualdl --logdir ./ --port 8080
Then, enter the address in the local browser: http://localhost:8080/, and you can access it.
```

The example is as follows, see `Indicator Trend Map.png`

![](03-ac592509.png)

4. Begin the assessment (important ⭐⭐⭐)

**Operations in the PaddleX source code directory**

Create a new shell and run the script `script_run_eval.sh` with the following content:

```
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./datasets/mmafedb \
    -o Evaluate.weight_path=./output/best_model/best_model.pdparams \
    -o Evaluate.log_interval=10
```

PS: When evaluating the model, it is necessary to specify the path of the model weight file.

After model evaluation, `evaluate_result.json` will be generated in the `./output directory`, which records the evaluation result of the evaluation task, specifically whether the evaluation task is completed normally and the evaluation metrics of the model, including val.top1 and val.top5, as follows:

```
{
  "done_flag": true,
  "metrics": {
    "val.top1": 0.57093,
    "val.top5": 0.927
  }
}
```

5. Starting the reasoning (important ⭐⭐⭐)

**Operations in the PaddleX source code directory**

Create a new shell and run the script `script_run_predict.sh` with the following content:

```
python main.py -c paddlex/configs/modules/image_classification/PP-LCNet_x1_0.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="./test_imgs/test_001.png"
```

PS: When model inference is required, the model weight folder path needs to be specified, including the *.pdiparams file of the model in inference mode.

After completing the inference and prediction, `test_001_res.json` will be generated in the `./output directory`, which records the `predicted results`. Specifically, it records whether the prediction task was completed normally, as well as the category scores, etc. The recognition result image is shown below.

![](04-ea62bb49.png)

**(Additional) Model Integration and Inference**

Create a new code file named `fast_predict.py` with the following content:

```
from paddlex import create_model
model_dir = "./output/best_model/inference"
model = create_model(model_name="PP-LCNet_x1_0", model_dir=model_dir)
output = model.predict("./test_imgs", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./fast_preict_output/")
    res.save_to_json("./fast_preict_output/res.json")
```

The results are saved to the `./fast_preict_output` directory.

6. Reference resource links

-

• https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/image_classification.html

-

• https://paddlepaddle.github.io/PaddleX/latest/module_usage/instructions/config_parameters_common.html

7. Exchange and Learning

Continue to exchange and learn, promoting mutual progress for you and me. Feel free to **reply and contact** below! Wishing you a prosperous future!


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
