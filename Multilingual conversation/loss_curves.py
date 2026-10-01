import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

######修改参数########
## 在这里输入check_point文件夹位置
checkpoint_path = "fine_tune_checkpoints/mt5_finetune"
# 设定 Batch Size (用于横轴缩放)
train_batch_size = 16 / 2
# 设定平滑窗口大小 (仅用于训练 Loss)
SMOOTHING_WINDOW = 30
#图片保存位置
save_path='loss_curves.png'
######修改参数（完）########


with open(checkpoint_path + "/training_log_history.pkl", "rb") as f:
    log_history = pickle.load(f)

# 2. 转换为 DataFrame
df = pd.DataFrame(log_history)

train_df = df.dropna(subset=['loss']).copy()  # 训练 Loss
eval_df = df.dropna(subset=['eval_loss']).copy()  # 评估数据 (包含 eval_loss 和 eval_bleu)

print("原始训练步数:",max(train_df['step']))

# 4. 数据预处理和缩放
# --- 4A. 横轴缩放 ---
# 逻辑：将每一步的 step 乘以 batch_size
if not train_df.empty:
    train_df['scaled_step'] = train_df['step'] * train_batch_size
if not eval_df.empty:
    eval_df['scaled_step'] = eval_df['step'] * train_batch_size

# --- 4B. 训练 Loss 平滑处理 ---
if not train_df.empty:
    # 应用指数加权移动平均 (EWMA)
    train_df['smoothed_loss'] = train_df['loss'].ewm(span=SMOOTHING_WINDOW, adjust=False).mean()

# 5. 绘图：创建包含四个子图的画布
# ax1 用于 Loss，ax2 用于 BLEU
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)  # sharex=True 确保横轴一致
fig,(ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)  # 增加到四个子图


# --- 子图 1: Loss 曲线 ---
ax1.set_title('Training and Evaluation Loss Curve (Smoothed)')
ax1.set_ylabel('Loss')

# 绘制平滑后的训练 Loss
if not train_df.empty:
    ax1.plot(train_df['scaled_step'], train_df['smoothed_loss'],
             label=f'Smoothed Training Loss (EWMA, span={SMOOTHING_WINDOW})',
             linestyle='-',
             color='blue',
             alpha=0.8)

# 绘制评估 Loss
if not eval_df.empty:
    ax1.plot(eval_df['scaled_step'], eval_df['eval_loss'],
             label='Evaluation Loss (Checkpoint)',
             marker='o',
             linestyle='-',
             color='red',
             alpha=0.9)

ax1.legend()
ax1.grid(True)

# --- 子图 2: BLEU 指标曲线 ---
ax2.set_title('Evaluation BLEU Scores')
#与后续公用x轴
ax2.set_ylabel('BLEU Score')
if not eval_df.empty:
    # 绘制 eval_bleu (总分数)
    ax2.plot(eval_df['scaled_step'], eval_df['eval_bleu'],
             label='Total BLEU Score',
             marker='o',
             linestyle='-',
             color='green',
             linewidth=2)

    # 绘制 eval_bleu-1 (1-gram 精度)
    ax2.plot(eval_df['scaled_step'], eval_df['eval_bleu-1'],
             label='BLEU-1 (Unigram Precision)',
             marker='^',
             linestyle='--',
             color='orange')

    # 绘制 eval_bleu-2 (2-gram 精度)
    ax2.plot(eval_df['scaled_step'], eval_df['eval_bleu-2'],
             label='BLEU-2 (Bigram Precision)',
             marker='s',
             linestyle=':',
             color='purple')

ax2.legend()
ax2.grid(True)

# --- 子图 3: RoUGE-L 指标曲线 ---
ax3.set_title('Evaluation ROUGE-L Scores')
ax3.set_ylabel('ROUGE-L Score')
if not eval_df.empty:
    # 绘制 eval_rouge1 (ROUGE-1 分数)
    ax3.plot(eval_df['scaled_step'], eval_df['eval_rouge1'],
             label='ROUGE-1 Score',
             marker='^',
             linestyle='--',
             color='orange')

    # 绘制 eval_rouge2 (ROUGE-2 分数)
    ax3.plot(eval_df['scaled_step'], eval_df['eval_rouge2'],
             label='ROUGE-2 Score',
             marker='s',
             linestyle=':',
             color='purple')

    # 绘制 eval_rougeL (ROUGE-L 分数)
    ax3.plot(eval_df['scaled_step'], eval_df['eval_rougeL'],
             label='ROUGE-L Score',
             marker='o',
             linestyle='-',
             color='green',
             linewidth=2)

ax3.legend()
ax3.grid(True)

# --- 子图 4: Distinct-1 和 Distinct-2 指标曲线 ---
ax4.set_title('Evaluation Distinct-1 and Distinct-2 Scores')
ax4.set_xlabel(f'Effective Training Step (Scaled by Batch Size {train_batch_size})')
ax4.set_ylabel('Distinct Scores')
if not eval_df.empty:
    # 绘制 eval_distinct-1
    ax4.plot(eval_df['scaled_step'], eval_df['eval_distinct-1'],
             label='Distinct-1 Score',
             marker='^',
             linestyle='--',
             color='orange')

    # 绘制 eval_distinct-2
    ax4.plot(eval_df['scaled_step'], eval_df['eval_distinct-2'],
             label='Distinct-2 Score',
             marker='s',
             linestyle=':',
             color='purple')
ax4.legend()
ax4.grid(True)
# 若要增加子图，复制上面的代码改一改即可

# 调整子图间距
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()