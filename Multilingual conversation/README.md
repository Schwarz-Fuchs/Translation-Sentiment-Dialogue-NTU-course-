# 极简使用说明

#### 1.运行 [parser_notationed.py](parser_notationed.py) 生成./data/raw 下的原始.src,tgt文件 （代码只保留了中英文）

#### 2.运行 [preprocess_notationed.py](preprocess_notationed.py) 生成./data 下三种任务的.json文件

#### 3.给 [finetune_notationed.py](finetune_notationed.py) 设置运行参数

参数位置为

```python
 ##缓存位置，如果需要腾硬盘空间可以清理
    cache_dir='.\cache'

    ##ZY 251104 增加功能 从检查点接着训练 （注意！！ 请在从更新文件前备份自己训练时设置的参数）
    resume_training_from_checkpoint = None  #不识别检查点，直接从头训练
    #resume_training_from_checkpoint = True  #识别最后一个检查点继续训练
    #resume_training_from_checkpoint = "./fine_tune_checkpoints/mt5_finetune/checkpoint-10000" #从某个特定检查点继续训练

    ## 你可以在CMD/bash用huggingface-cli下载，也可以直接把模型名称填到model_name_or_path里面
    ## huggingface-cli download google/mt5-base --local-dir ./models
    
   ########## 需要修改的参数 #########
    model_args = ModelArguments(
        model_name_or_path="models/mt5-base",  # 替换为您想使用的预训练模型路径, 如果输入huggingface模型名称下载的话，它会保存在
                                               # C:\Users\你的用户名\.cache\huggingface\hub
    )

    data_args = DataTrainingArguments(
        data_path="data/",  # 数据位置
        data_name="multilingual",  # 具体的任务
        train_file="train.jsonl",   #最终的训练集位置会被拼接为 data_path/data_name/train_file
        validation_file="dev.jsonl",
        max_source_length=512,
        max_target_length=256,
        val_max_target_length=256,
        num_beams=5,
        preprocessing_num_workers=4,
        length_penalty=1.5, #251105新增 长度惩罚 防止模型生成保守的过短的序列
        no_repeat_ngram_size=3,  #251105新增 重复生成惩罚 防止模型说车轱辘话
        use_slow_tokenizer=True  #251105新增 可以使用慢速编码器来提高模型效果，避免生成大量的 <unk> 标记
        ##调试用参数，只用于测试代码能不能跑 正常训练时请把这三个注释掉 否则会梯度爆炸
        #max_train_samples=10,
        #max_eval_samples=10,
        #max_predict_samples=10,
    )

    #不想设置的可以注释掉，会使用默认值
    training_args = Seq2SeqTrainingArguments(
        output_dir="./fine_tune_checkpoints/mt5_finetune",  # !!! 替换为模型保存路径
        do_train=True,
        do_eval=True,
        num_train_epochs=1,
        max_steps=10000,  #最大步数，到此步会停止训练，如果不需要最大步数请注释掉 测试代码我放的很小

        per_device_train_batch_size=8,
        gradient_accumulation_steps=2, #用gradient_accumulation_steps获得等效8*2=16的训练batch, 但会降低一定训练速度 use at your own risk
                                       #另外如果gradient_accumulation_steps=2 相当于模型跑了2步再进行更新 实际有效步数为max_steps/2
        per_device_eval_batch_size=32,

        learning_rate=1e-5,  #初始学习率
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        optim='adamw_torch',
        max_grad_norm=1.0, #设置梯度上限防止梯度爆炸

        save_strategy="steps", #也可以用epoch
        save_steps=2000,
        eval_strategy="steps",  #在部分 Seq2SeqTrainingArguments 版本中，这里可能需要修改为 evaluation_strategy
        eval_steps=1000,
        logging_steps=10,  #多少step输出一次训练状况（这个值会保留在.pkl中绘图）
        load_best_model_at_end=True,  #评估表现最好的模型
        metric_for_best_model="bleu",  #评估依据  ,默认是loss
        greater_is_better=True,        #metric_for_best_model 是值越大越好 (如 BLEU) 还是越小越好 (如 Loss)
        predict_with_generate=True,
        prediction_loss_only=False,
        #early_stopping_patience=5,
        fp16=False,  #fp16可以加速训练，但可能导致训练不稳定，use it at your own risk
        seed=42,
        report_to="none" # 若要启用日志记录器，请使用cmd运行，并输入你的API密钥，比如  wandb.login(key="[您的 API 密钥]")
    )
    ########## 需要修改的参数 (完) #########
```

#### 4.运行[finetune_notationed.py](finetune_notationed.py)
从我个人使用 google/mt5-small 的测试经验看，如果设置的per_device_train_batch_size太小，一开始很容易出现梯度爆炸现象。

如果你发现训练log里面 loss 是0 grad_norm 是 nan 那么就是梯度爆炸了， 如下例

```{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 4.7184665678887554e-06, 'epoch': 0.36}```

我在默认设置里增加了max_grad_norm=1.0和warmup_ratio=0.1, 以减轻爆炸影响，你也可以通过增大per_device_train_batch_size来获得更稳定的梯度，如果你的现存能吃得住的话

我测试的极端样例 训练batchsize只有4的时候，花了5000步才把'grad_norm'（梯度的norm）从上万拉到10左右, 总之越小的batchsize 就需要越小的LR和越长的训练步数

至于需要训练多少步，还在摸索，大概是用3到5 乘以 Steps per Epoch
$$\text{Steps per Epoch} = \frac{\text{训练样本总数}}{\text{Batch Size}}$$


#### 5.[可选项] 运行[loss_curves.py](loss_curves.py) 来画各种图
```python
######修改参数########
## 在这里输入check_point文件夹位置，里面应该有个.pkl文件
checkpoint_path = "fine_tune_checkpoints/mt5_finetune"
# 设定 Batch Size (用于横轴缩放)
train_batch_size = 16
# 设定平滑窗口大小 (仅用于训练 Loss)
SMOOTHING_WINDOW = 30
#图片保存位置
save_path='loss_curves.png'
######修改参数（完）########
```
#### 6.[可选项] 运行[inference.py](inference.py) 来测试对话
```python
 # 替换为检查点目录路径。
    CHECKPOINT_PATH = "./fine_tune_checkpoints/mt5_finetune/checkpoint-10000"
    #预训练模型位置（被微调的模型文件）
    BASE_MODEL_NAME = "models/mt5-base" 
    # 测试文本
    INPUT_TEXTS=['hello','what a good day today','早上好','今天天气真不错']
```


##### 备注：Warning我没有管，会报的很热闹 (11月4日更新: Warning 应该不会狠狠报了)
##### 备注:代码相较原始版本进行了一定修改，比如取消了原本dataset.py的使用。直接读json文件。代码的测试环境见requirements.txt，我没有为这个项目专门设置虚拟环境，所以有不少没用的，仅供包冲突时参考