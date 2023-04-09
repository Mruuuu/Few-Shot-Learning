<!--
 * @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 * @Date: 2023-03-23 21:22:33
 * @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 * @LastEditTime: 2023-04-09 23:48:04
 * @FilePath: /mru/Few-Shot-Learning/README.md
 * @Description: 
 * 
-->
# Few-Shot-Learning
**2023 Spring, Data Science HW3**

[\[Spec v1\]](./assests/hw3_few_shot_learning.pdf), [\[Lecture\]](./assests/Lecture4-Few-Shot-Learning.pdf), [\[Kaggle\]](https://www.kaggle.com/competitions/data-science-2023-hw3-few-shot-learning/overview)

<br/>

## Data preparing
- Download the data from Kaggle
    ```
    kaggle competitions download -c data-science-2023-hw3-few-shot-learning
    ```

- Unzip the data into `./`
    ```
    unzip data-science-2023-hw3-few-shot-learning.zip
    ```

<br/>

## Folder structure
```
.
├── assets/
│   ├── hw3_few_shot_learning.pdf
│   └── Lecture4-Few-Shot-Learning.pdf
├── helper/
│   ├── visualization.ipynb
│   ├── view_performance.py
│   ├── parameters.py
│   └── utils.py
├── release/
│   ├── test.pkl
│   ├── train.pkl
│   └── validation.pkl
├── scripts/
│   └── [some hyperparmeter tuning .sh]
├── train/
│   ├── dataloader.py
│   ├── models.py
│   └── train.py
├── .gitignore
├── run.sh
├── Readme.md
├── requirements.txt
└── main.py
```
Note: Make sure that the data file `release/` is at the path `./`

<!-- Note: The well-trained pruned model can be found at `./assests` -->

<br/>

## Environment setup
- Python 3.8.10
    ```sh
    pip3 install -r requirements.txt
    ```
- Or
    ```
    pip3 install torch torchvision torchsummary tensorboard tqdm datetime pyyaml pandas kaggle jupyter termcolor numpy
    ```
    
<br/>

## Run code
```sh
chmod +x ./run.sh
./run.sh
```
Note: After running the script, the prediction file `pred.csv` can be found at `./logs/test/pred.csv`

<br/>

## Hyperparmeter tuning (optional)
```sh
chmod +x ./scripts/{file_name}.sh
./scripts/{file_name}.sh
```
Note: one can modify this file to tune any parameter

Note: one can modify and run `./helper/view_performance.py` to see the tuning output 

<br/>

## Referenced
- https://github.com/sicara/easy-few-shot-learning
- https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py