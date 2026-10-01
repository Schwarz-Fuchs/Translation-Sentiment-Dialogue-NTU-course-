import os
import shutil


############生成训练用数据集 .src .tgt 格式（Dataloader 可读取） 生成结果在./data/raw #############

def generate_data_for_monolingual(input_filename, output_filename, target_filename, flag, language_flag):
    # 定义函数：为单语言任务(monolingual)生成数据集。

    def not_empty(s):
        # 定义内部辅助函数：检查字符串是否非空且非全空白。
        return s and s.strip()

    output_file = open(output_filename, 'w')
    # 以写入模式打开源文件（output_filename）。
    target_file = open(target_filename, 'w')
    # 以写入模式打开目标文件（target_filename）。
    for line in open(input_filename, 'r').readlines():
        # 遍历输入文件中的每一行。
        lineArr = line.strip().split('\t')
        # 去除行首尾空白，并按制表符（\t）分割成数组。

        # data = lineArr[0].strip()[::-1].replace('__uoe__', '', 1)[::-1].strip().split('__eou__')
        # 处理特定标记（__uoe__）的旧逻辑。

        data = lineArr[0].strip().split('__eou__')
        # 取第一列（对话内容），去除首尾空白，按对话单元结束标记（__eou__）分割。
        data = list(filter(not_empty, data))
        # 过滤掉列表中空字符串或全空白的元素。
        for i in range(1, len(data) - 1, 2):
            # 遍历对话回合（从索引 1 开始，每隔 2 个元素，跳过最后一个可能的空元素）。
            context = data[:i]
            # 上下文：从开头到当前回复之前的所有对话单元。
            response = data[i].strip()
            # 响应：当前的对话单元。
            current_tae = [lineArr[1][0], lineArr[2].strip().split(' ')[i], lineArr[3].strip().split(' ')[i]]
            # 获取当前对话回合的主题（T）、行为（A）、情感（E），其中 i 用于索引。
            output_file.write(' '.join(current_tae) + '</s>' + '</s>'.join(context) + ' ' + language_flag + '\n')
            # 写入源文件：[T A E]</s>[上下文对话单元1]</s>[上下文对话单元2]... [语言标记]\n。
            target_file.write(response + '\n')
            # 写入目标文件：[响应]\n。
    output_file.close()
    # 关闭源文件。
    target_file.close()
    # 关闭目标文件。


def generate_data_for_multilingual(input_filename, input_target_filename, output_filename, target_filename):
    # 定义函数：为多语言任务生成数据（通过追加方式合并数据）。
    output_file = open(output_filename, 'a')
    # 以追加模式打开源输出文件。
    for line in open(input_filename, 'r').readlines():
        # 遍历输入的源文件。
        output_file.write(line)
        # 将内容追加写入源输出文件。
    output_file.close()
    # 关闭源输出文件。

    target_file = open(target_filename, 'a')
    # 以追加模式打开目标输出文件。
    for line in open(input_target_filename, 'r').readlines():
        # 遍历输入的目标文件。
        target_file.write(line)
        # 将内容追加写入目标输出文件。
    target_file.close()
    # 关闭目标输出文件。


def get_utterance_pair(en_input_filename, other_input_filename, output_filename):
    # 定义函数：获取（英文与其他语言的）对话单元对。
    en_utterance = []
    # 初始化英文对话单元列表。
    other_utterance = []
    # 初始化其他语言对话单元列表。

    other_input_file_name = other_input_filename
    en_input_file_name = en_input_filename
    # 临时保存文件名。

    en_input_file_list = [en_input_filename, en_input_file_name.replace('train', 'dev'),
                          en_input_file_name.replace('train', 'test')]
    # 创建英文文件的列表：包含 train, dev, test 三个文件路径。
    other_input_file_list = [other_input_filename, other_input_file_name.replace('train', 'dev'),
                             other_input_file_name.replace('train', 'test')]
    # 创建其他语言文件的列表：包含 train, dev, test 三个文件路径。
    for en_input_filename in en_input_file_list:
        # 遍历所有英文输入文件。
        for line in open(en_input_filename, 'r').readlines():
            # 遍历文件中的每一行。
            lineArr = line.strip().split('__eou__')
            # 按对话单元结束标记（__eou__）分割行。
            for la in lineArr:
                en_utterance.append(la.strip())
                # 将每个对话单元去除空白后添加到英文列表。
    for other_input_filename in other_input_file_list:
        # 遍历所有其他语言的输入文件。
        for line in open(other_input_filename, 'r').readlines():
            # 遍历文件中的每一行。
            lineArr = line.strip().split('__eou__')
            # 按对话单元结束标记（__eou__）分割行。
            for la in lineArr:
                other_utterance.append(la.strip())
                # 将每个对话单元去除空白后添加到其他语言列表。
    if len(en_utterance) == len(other_utterance):
        # 检查两个列表的长度是否相等（确保对话单元对齐）。
        output_file = open(output_filename, 'w')
        # 以写入模式打开输出文件。
        for i in range(len(en_utterance)):
            # 遍历所有对话单元。
            output_file.write(en_utterance[i] + '\t' + other_utterance[i] + '\n')
            # 将英文对话单元和其他语言对话单元用制表符连接并写入文件。
        output_file.close()
        # 关闭输出文件。
    else:
        print('Data Error!')
        # 如果长度不一致，打印“数据错误！”。


def generate_data_for_crosslingual(input_src_filename, language_flag, dict_filename, output_src_filename,
                                   output_tgt_filename, flag):
    # 定义函数：为跨语言任务生成数据。
    def not_empty(s):
        # 定义内部辅助函数：检查字符串是否非空且非全空白。
        return s and s.strip()

    current_dict = {}
    # 初始化一个字典用于存储对话单元的跨语言映射。
    for line in open(dict_filename, 'r').readlines():
        # 遍历字典文件（包含对话单元对齐信息）。
        lineArr = line.strip().split('\t')
        # 按制表符分割行。
        if len(lineArr) == 2:
            # 如果行包含两个元素（即一对对齐的对话单元）。
            if flag == 'En':
                current_dict[lineArr[0].strip()] = lineArr[1].strip()
                # 如果目标是英文（flag='En'），则将源语言作为键，英文作为值。
            else:
                current_dict[lineArr[1].strip()] = lineArr[0].strip()
                # 如果目标不是英文（flag!='En'），则将英文作为键，目标语言作为值。

    output_file = open(output_src_filename, 'w')
    # 以写入模式打开源输出文件。
    output_tgt_file = open(output_tgt_filename, 'w')
    # 以写入模式打开目标输出文件。
    for line in open(input_src_filename, 'r').readlines():
        # 遍历输入的源文件（包含原始对话数据）。
        lineArr = line.strip().split('\t')
        # 按制表符分割行。
        # data = lineArr[0].strip()[::-1].replace('__uoe__', '', 1)[::-1].strip().split('__eou__')
        # 注释掉的旧逻辑。
        data = lineArr[0].strip().split('__eou__')
        # 取对话内容，按__eou__分割。
        data = list(filter(not_empty, data))
        # 过滤掉空字符串。
        for i in range(1, len(data), 2):
            # 遍历对话回合（响应）。
            context = data[:i]
            # 上下文：当前响应前的所有对话单元。
            response = data[i].strip()
            # 响应。
            crosslingual_context = []
            # 初始化跨语言上下文列表。
            for c in context:
                crosslingual_context.append(current_dict[c.strip()])
                # 查找上下文中的每个对话单元在字典中的对应跨语言版本。
            current_tae = [lineArr[1][0], lineArr[2].strip().split(' ')[i], lineArr[3].strip().split(' ')[i]]
            # 获取当前回合的 T, A, E 标签。
            # print(current_tae, crosslingual_context, response, i, len(current_tae))
            # 注释掉的调试打印。
            output_file.write(
                ' '.join(current_tae) + '</s>' + '</s>'.join(crosslingual_context) + ' ' + language_flag + '\n')
            # 写入源文件：[T A E]</s>[跨语言上下文单元1]</s>[...]\n。
            output_tgt_file.write(response + '\n')
            # 写入目标文件：[响应]\n。
    output_file.close()
    # 关闭源文件。
    output_tgt_file.close()
    # 关闭目标文件。


def get_dialog_dict(en2a_filename, en2b_filename):  # a2en + en2b -->a2b
    # 定义函数：通过英文（En）作为中介，获取从语言 A 到语言 B 的对话单元映射。
    a2en = {}
    # 初始化 A 到 En 的字典。
    for line in open(en2a_filename, 'r').readlines():
        # 遍历 A2En 文件（实际存储的是 En2A 的对齐，但这里是用于构建 A2En 的映射）。
        lineArr = line.strip().split('\t')
        if len(lineArr) == 2:
            a2en[lineArr[1].strip()] = lineArr[0].strip()
            # 假设文件格式是 En\tA，则建立 A -> En 的映射。
    en2b = {}
    # 初始化 En 到 B 的字典。
    for line in open(en2b_filename, 'r').readlines():
        # 遍历 En2B 文件。
        lineArr = line.strip().split('\t')
        if len(lineArr) == 2:
            en2b[lineArr[0].strip()] = lineArr[1].strip()
            # 建立 En -> B 的映射。
    a2b = {}
    # 初始化 A 到 B 的字典。
    for k in a2en:
        # 遍历 A 中的对话单元。
        a2b[k] = en2b[a2en[k]]
        # 链式查找：A -> En，然后 En -> B，建立 A -> B 的映射。
        # print(k + '\t' + en2b[a2en[k]])
        # 注释掉的调试打印。
    return a2b
    # 返回 A 到 B 的映射字典。


def generate_data_for_crosslingual_no_En(input_src_filename, language_flag, en2a_filename, en2b_filename,
                                         output_src_filename, output_tgt_filename):
    # 定义函数：为不涉及英文的跨语言任务生成数据（例如从德语到中文）。
    current_dict = get_dialog_dict(en2a_filename, en2b_filename)
    # 调用 get_dialog_dict 函数获取 A 到 B 的映射字典。

    output_file = open(output_src_filename, 'w')
    # 以写入模式打开源输出文件。
    output_tgt_file = open(output_tgt_filename, 'w')
    # 以写入模式打开目标输出文件。
    for line in open(input_src_filename, 'r').readlines():
        # 遍历输入的源文件。
        lineArr = line.strip().split('\t')
        # 按制表符分割行。
        # data = lineArr[0].strip()[::-1].replace('__uoe__', '', 1)[::-1].strip().split('__eou__')
        # 注释掉的旧逻辑。
        data = lineArr[0].strip().split('__eou__')
        # 取对话内容，按__eou__分割。
        for i in range(1, len(data), 2):
            # 遍历对话回合（响应）。
            context = data[:i]
            # 上下文。
            response = data[i].strip()
            # 响应。
            crosslingual_context = []
            # 初始化跨语言上下文列表。
            for c in context:
                crosslingual_context.append(current_dict[c.strip()])
                # 查找上下文中的每个对话单元在 A2B 字典中的对应跨语言版本。
            current_tae = [lineArr[1][0], lineArr[2].strip().split(' ')[i], lineArr[3].strip().split(' ')[i]]
            # 获取当前回合的 T, A, E 标签。
            # print(current_tae, crosslingual_context, response, i, len(current_tae))
            # 注释掉的调试打印。
            output_file.write(
                ' '.join(current_tae) + '</s>' + '</s>'.join(crosslingual_context) + ' ' + language_flag + '\n')
            # 写入源文件：[T A E]</s>[A2B 跨语言上下文单元1]</s>[...]\n。
            output_tgt_file.write(response + '\n')
            # 写入目标文件：[响应]\n。
    output_file.close()
    # 关闭源文件。
    output_tgt_file.close()
    # 关闭目标文件。


def mkdirs(root):
    # 定义函数：创建所需的数据目录结构。
    try:
        # 尝试创建目录。
        os.mkdir(os.path.join(root, "multilingual"))
        # 创建 multilingual 目录。
        lans = ["En", "Zh"]
        #lans = ["En", "Zh", "De", "It"]
        # 定义单语言列表。
        os.mkdir(os.path.join(root, "monolingual"))
        # 创建 monolingual 目录。
        for lan in lans:
            os.mkdir(os.path.join(root, "monolingual", lan))
            # 在 monolingual 下为每种语言创建子目录。
        lans = ["En_Zh", "Zh_En"]
        #lans = ["En_Zh", "Zh_En", "En_De", "De_En"]
        # 定义跨语言列表（源_目标）。
        os.mkdir(os.path.join(root, "crosslingual"))
        # 创建 crosslingual 目录。
        for lan in lans:
            os.mkdir(os.path.join(root, "crosslingual", lan))
            # 在 crosslingual 下为每种语言对创建子目录。
    except:
        # 如果目录已存在或创建失败（如权限问题），则忽略。
        pass


if __name__ == '__main__':
    # 主执行块：当脚本作为主程序运行时执行。
    mkdirs("./data")

    # 标记：单语言数据处理开始。
    generate_data_for_monolingual('data/en_train_human.txt', 'data/monolingual/En/train.src',
                                  'data/monolingual/En/train.tgt', 1, '<En>')
    # 为英文训练集生成单语言数据。
    generate_data_for_monolingual('data/zh_train_human.txt', 'data/monolingual/Zh/train.src',
                                  'data/monolingual/Zh/train.tgt', 1, '<Zh>')
    # 为中文训练集生成单语言数据。

    '''
    generate_data_for_monolingual('data/de_train_human.txt', 'data/monolingual/De/train.src','data/monolingual/De/train.tgt', 1, '<De>')
    # 为德语训练集生成单语言数据。
    generate_data_for_monolingual('data/it_train_human.txt', 'data/monolingual/It/train.src','data/monolingual/It/train.tgt', 1, '<It>')
    # 为意大利语训练集生成单语言数据。
    '''

    generate_data_for_monolingual('data/en_dev_human.txt', 'data/monolingual/En/dev.src', 'data/monolingual/En/dev.tgt',
                                 1, '<En>')
    # 为英文验证集生成单语言数据。
    generate_data_for_monolingual('data/zh_dev_human.txt', 'data/monolingual/Zh/dev.src', 'data/monolingual/Zh/dev.tgt',
                                  1, '<Zh>')
    # 为中文验证集生成单语言数据。
    '''
    generate_data_for_monolingual('data/de_dev_human.txt', 'data/monolingual/De/dev.src', 'data/monolingual/De/dev.tgt',1, '<De>')
    # 为德语验证集生成单语言数据。
    generate_data_for_monolingual('data/it_dev_human.txt', 'data/monolingual/It/dev.src', 'data/monolingual/It/dev.tgt',1, '<It>')
    # 为意大利语验证集生成单语言数据。
    '''

    generate_data_for_monolingual('data/en_test_human.txt', 'data/monolingual/En/test.src',
                                  'data/monolingual/En/test.tgt', 1, '<En>')
    # 为英文测试集生成单语言数据。
    generate_data_for_monolingual('data/zh_test_human.txt', 'data/monolingual/Zh/test.src',
                                  'data/monolingual/Zh/test.tgt', 1, '<Zh>')
    # 为中文测试集生成单语言数据。
    '''
    generate_data_for_monolingual('data/de_test_human.txt', 'data/monolingual/De/test.src','data/monolingual/De/test.tgt', 1, '<De>')
    # 为德语测试集生成单语言数据。
    generate_data_for_monolingual('data/it_test_human.txt', 'data/monolingual/It/test.src','data/monolingual/It/test.tgt', 1, '<It>')
    # 为意大利语测试集生成单语言数据。
    '''

    # multilingual
    # 标记：多语言数据处理开始。
    try:
        # 删除旧的多语言训练集文件。
        os.remove("data/multilingual/train.src")
        os.remove("data/multilingual/train.tgt")
    except:
        # 如果文件不存在，则忽略错误。
        pass

    '''
    generate_data_for_multilingual('data/monolingual/De/train.src', 'data/monolingual/De/train.tgt',
                                   'data/multilingual/train.src', 'data/multilingual/train.tgt')
    # 将德语训练集数据追加到多语言训练集。
    generate_data_for_multilingual('data/monolingual/It/train.src', 'data/monolingual/It/train.tgt',
                                   'data/multilingual/train.src', 'data/multilingual/train.tgt')
    # 将意大利语训练集数据追加到多语言训练集。
    '''
    generate_data_for_multilingual('data/monolingual/En/train.src', 'data/monolingual/En/train.tgt',
                                   'data/multilingual/train.src', 'data/multilingual/train.tgt')

    # 将英文训练集数据追加到多语言训练集。
    generate_data_for_multilingual('data/monolingual/Zh/train.src', 'data/monolingual/Zh/train.tgt',
                                   'data/multilingual/train.src', 'data/multilingual/train.tgt')
    # 将中文训练集数据追加到多语言训练集。

    '''
    generate_data_for_multilingual('data/monolingual/De/dev.src', 'data/monolingual/De/dev.tgt',
                                   'data/multilingual/dev.src', 'data/multilingual/dev.tgt')
    # 将德语验证集数据追加到多语言验证集。
    generate_data_for_multilingual('data/monolingual/It/dev.src', 'data/monolingual/It/dev.tgt',
                                   'data/multilingual/dev.src', 'data/multilingual/dev.tgt')
    # 将意大利语验证集数据追加到多语言验证集。
    '''

    generate_data_for_multilingual('data/monolingual/En/dev.src', 'data/monolingual/En/dev.tgt',
                                   'data/multilingual/dev.src', 'data/multilingual/dev.tgt')
    # 将英文验证集数据追加到多语言验证集。

    generate_data_for_multilingual('data/monolingual/Zh/dev.src', 'data/monolingual/Zh/dev.tgt',
                                   'data/multilingual/dev.src', 'data/multilingual/dev.tgt')
    # 将中文验证集数据追加到多语言验证集。

    '''
    generate_data_for_multilingual('data/monolingual/De/test.src', 'data/monolingual/De/test.tgt',
                                   'data/multilingual/test.src', 'data/multilingual/test.tgt')
    # 将德语测试集数据追加到多语言测试集。
    generate_data_for_multilingual('data/monolingual/It/test.src', 'data/monolingual/It/test.tgt',
                                   'data/multilingual/test.src', 'data/multilingual/test.tgt')
    # 将意大利语测试集数据追加到多语言测试集。
    '''

    generate_data_for_multilingual('data/monolingual/En/test.src', 'data/monolingual/En/test.tgt',
                                   'data/multilingual/test.src', 'data/multilingual/test.tgt')
    # 将英文测试集数据追加到多语言测试集。

    generate_data_for_multilingual('data/monolingual/Zh/test.src', 'data/monolingual/Zh/test.tgt',
                                   'data/multilingual/test.src', 'data/multilingual/test.tgt')
    # 将中文测试集数据追加到多语言测试集。


    # crosslingual
    # 标记：跨语言数据处理开始。
    get_utterance_pair('data/en_train_human.txt', 'data/zh_train_human.txt', 'data/En2Zh.txt')
    # 获取英文和中文训练集对话单元对，保存为 En2Zh.txt（用于构建跨语言字典）。
    '''
    get_utterance_pair('data/en_train_human.txt', 'data/de_train_human.txt', 'data/En2De.txt')
    # 获取英文和德语训练集对话单元对，保存为 En2De.txt。
    get_utterance_pair('data/en_train_human.txt', 'data/it_train_human.txt', 'data/En2It.txt')
    # 获取英文和意大利语训练集对话单元对，保存为 En2It.txt。
    '''


    generate_data_for_crosslingual('data/zh_train_human.txt', '<Zh>', 'data/En2Zh.txt',
                                   'data/crosslingual/En_Zh/train.src', 'data/crosslingual/En_Zh/train.tgt', 'Zh')
    # 生成 En_Zh（英文上下文，目标中文）的训练集。
    generate_data_for_crosslingual('data/en_train_human.txt', '<En>', 'data/En2Zh.txt',
                                   'data/crosslingual/Zh_En/train.src', 'data/crosslingual/Zh_En/train.tgt', 'En')
    # 生成 Zh_En（中文上下文，目标英文）的训练集。
    '''
    generate_data_for_crosslingual('data/de_train_human.txt', '<De>', 'data/En2De.txt',
                                   'data/crosslingual/En_De/train.src', 'data/crosslingual/En_De/train.tgt', 'De')
    # 生成 En_De（英文上下文，目标德语）的训练集。
    generate_data_for_crosslingual('data/en_train_human.txt', '<En>', 'data/En2De.txt',
                                   'data/crosslingual/De_En/train.src', 'data/crosslingual/De_En/train.tgt', 'En')
    # 生成 De_En（德语上下文，目标英文）的训练集。
    '''
    generate_data_for_crosslingual('data/zh_dev_human.txt', '<Zh>', 'data/En2Zh.txt', 'data/crosslingual/En_Zh/dev.src',
                                   'data/crosslingual/En_Zh/dev.tgt', 'Zh')
    # 生成 En_Zh 的验证集。
    generate_data_for_crosslingual('data/en_dev_human.txt', '<En>', 'data/En2Zh.txt', 'data/crosslingual/Zh_En/dev.src',
                                   'data/crosslingual/Zh_En/dev.tgt', 'En')
    # 生成 Zh_En 的验证集。
    '''
    generate_data_for_crosslingual('data/de_dev_human.txt', '<De>', 'data/En2De.txt', 'data/crosslingual/En_De/dev.src',
                                   'data/crosslingual/En_De/dev.tgt', 'De')
    # 生成 En_De 的验证集。
    generate_data_for_crosslingual('data/en_dev_human.txt', '<En>', 'data/En2De.txt', 'data/crosslingual/De_En/dev.src',
                                   'data/crosslingual/De_En/dev.tgt', 'En')
    # 生成 De_En 的验证集。
    '''
    generate_data_for_crosslingual('data/zh_test_human.txt', '<Zh>', 'data/En2Zh.txt',
                                   'data/crosslingual/En_Zh/test.src', 'data/crosslingual/En_Zh/test.tgt', 'Zh')
    # 生成 En_Zh 的测试集。
    generate_data_for_crosslingual('data/en_test_human.txt', '<En>', 'data/En2Zh.txt',
                                   'data/crosslingual/Zh_En/test.src', 'data/crosslingual/Zh_En/test.tgt', 'En')
    # 生成 Zh_En 的测试集。
    '''
    generate_data_for_crosslingual('data/de_test_human.txt', '<De>', 'data/En2De.txt',
                                   'data/crosslingual/En_De/test.src', 'data/crosslingual/En_De/test.tgt', 'De')
    # 生成 En_De 的测试集。
    generate_data_for_crosslingual('data/en_test_human.txt', '<En>', 'data/En2De.txt',
                                   'data/crosslingual/De_En/test.src', 'data/crosslingual/De_En/test.tgt', 'En')
    # 生成 De_En 的测试集。
    '''
    os.mkdir("./data/raw")

    # 最终数据会挪到 /data/raw 文件夹下
    shutil.move("./data/monolingual", "./data/raw/monolingual")
    shutil.move("./data/crosslingual", "./data/raw/crosslingual")
    shutil.move("./data/multilingual", "./data/raw/multilingual")
