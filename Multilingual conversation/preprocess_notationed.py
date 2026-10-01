import os
import json
from transformers import T5TokenizerFast

#######生成训练用数据集 .src .tgt 转 .json格式 生成结果在./data ######

# 'google/mt5-base' tokenizer 在这里可以切换其他 tokenizer
tokenizer = T5TokenizerFast.from_pretrained('google/mt5-base')
dtypes = ['train', 'dev', 'test'] # 定义数据类型（训练集、开发集、测试集）
data_dir = './data/raw/'          # 输入数据集位置
save_dir = './data'               # 输出数据集位置

datas = []
for setting in os.listdir(data_dir): # 遍历原始数据目录下的所有设置（如 'monolingual', 'crosslingual', 'multilingual'）
    if "multilingual" not in setting:
        for task in os.listdir(data_dir + setting): # 非多语言则遍历该设置下的具体任务（e.g.'En', 'De_En'）
            datas.append(setting + "/" + task)
    else: # 如果设置是 'multilingual'
        datas.append(setting) # 直接将 'multilingual' 添加到列表中

print("setttings: {}".format(datas))

for data in datas:
    print(data)
    for dtype in dtypes: # 遍历 'train', 'dev', 'test' 三种数据类型
        src_lens, tgt_lens = [], [] # 初始化源句子和目标句子长度列表
        # 构造源文件和目标文件的完整路径 (例如：./data/raw/crosslingual/De_En/train.src)
        src_file, tgt_file = os.path.join(data_dir, data, dtype + '.src'), os.path.join(data_dir, data, dtype + '.tgt')
        # 读取源文件和目标文件的所有行
        src_lines, tgt_lines = open(src_file).readlines(), open(tgt_file).readlines()

        examples = []
        for src_line, tgt_line in zip(src_lines, tgt_lines): # 逐行配对处理源文件和目标文件
            src, tgt = src_line.strip(), tgt_line.strip() # 去除行首尾的空白字符

            src_lens.append(len(tokenizer.tokenize(src))) # 使用分词器对源句子分词并记录分词后的长度
            tgt_lens.append(len(tokenizer.tokenize(tgt))) # 使用分词器对目标句子分词并记录分词后的长度

            dic = {'src': src, 'tgt': tgt} # 创建包含源句子和目标句子的字典
            examples.append(json.dumps(dic, ensure_ascii=False))

        out_dir = os.path.join(save_dir, data)
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, dtype + '.jsonl')
        with open(out_file, 'w') as f:
            f.write('\n'.join(examples)) # 将所有 JSON 字符串用换行符连接后写入文件（生成 .jsonl 文件）

        print(len(examples), dtype) # 打印处理的示例数量和数据类型

        src_lens.sort(reverse=True) # 对源句子长度列表进行降序排序
        tgt_lens.sort(reverse=True) # 对目标句子长度列表进行降序排序

        print(src_lens[:100]) # 检查数据分布
        print(tgt_lens[:100]) # 检查数据分布