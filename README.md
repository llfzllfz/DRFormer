# 引入文件说明
+ DNABERT
    + 源自论文
    + 用途：序列模型
+ UFOLD - post processing 
    + 源自论文
    + 用途：二级结构预测后处理

# 数据说明
+ CLIP
    + 用途：RBP任务
    + 来源：
+ GUE
    + 用途：验证大模型，7个任务29个数据集，长度从70-1000
    + 来源：DNABERT2
+ ARCHIVEII
    + 用途：二级结构任务
    + 来源：
+ bpRNA-1m
    + 用途：二级结构任务
    + 来源：RNA secondary structure prediction using deep learning with thermodynamic integration
    ```
    Sato K, Akiyama M, Sakakibara Y. RNA secondary structure prediction using deep learning with thermodynamic integration[J]. Nature communications, 2021, 12(1): 941.
    ```
+ TrainSetA && TestSetA && TestSetB
    + 用途：二级结构任务
    + 来源：RNA secondary structure prediction using deep learning with thermodynamic integration
    ```
    Sato K, Akiyama M, Sakakibara Y. RNA secondary structure prediction using deep learning with thermodynamic integration[J]. Nature communications, 2021, 12(1): 941.
    ```

# 模型


# Usage
1. 安装对应的库
```python
conda create -n DRFormer python=3.9
pip install torch torchvision torchaudio
python setup.py develop
pip install -r requirements.txt
```

2. 新建文件夹
```
mkdir models # to store the model pkl
mkdir bin # to store the feature file
mkdir log # to store the log file
```
3. get the feature
所有特征都在cpp_source文件夹下，如果在bin文件夹空，则使用
```shell
sh run_scripts/cpp_compile.sh
```
在bin下面会得到对应的文件

4. 划分对应的数据集(clip例)
```python
mkdir data
sh run_scripts/split_clip_data.sh
```


5. 下载预训练模型SWIN-RNA
```
https://drive.google.com/drive/folders/1WwTPg3_o4VZGE2Mw39KoyiZu-mTRF1Q2?usp=sharing
```

放到models文件夹下面

下载DNABERT3预训练模型

6. 训练模型

```
# CLIP: to run the DRFormer with layer 2 in one dataset
sh run_scripts/CLIP/AUH_K562/run.sh 
```

# Update

+ 2025.2.27 Update

