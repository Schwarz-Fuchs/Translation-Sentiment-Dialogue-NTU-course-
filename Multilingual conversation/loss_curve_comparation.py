import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置 Matplotlib 样式
plt.style.use('seaborn-v0_8-whitegrid')

######修改参数########
## 在这里输入check_point文件夹位置
pkl_path_mbart50 = "fine_tune_checkpoints/model_compareation/training_log_history (mbart50).pkl"
pkl_path_mbart25 = "fine_tune_checkpoints/model_compareation/training_log_history（mbart25）.pkl"
pkl_path_mt5_base = "fine_tune_checkpoints/model_compareation/training_log_history (mt5_base).pkl"
pkl_path_M2M100_1_2B = "fine_tune_checkpoints/model_compareation/training_log_history_1_2b.pkl"
pkl_path_M2M100_418M = "fine_tune_checkpoints/model_compareation/training_log_history(m2m100-418m).pkl"

# 设定 Batch Size (用于横轴缩放)
# 注意：train_batch_size = 16 / 2 表示 effective batch size 是 8
train_batch_size = 16 / 2
# 设定平滑窗口大小 (仅用于训练 Loss)
SMOOTHING_WINDOW = 30
# 图片保存目录
save_dir = 'model_comparison_plots'
# 图片的基础文件名
base_filename = 'model_comparison'
# *** 新增的起始横坐标设置 ***
START_X_AXIS = -10000
######修改参数（完）########

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# --- 1. 数据加载 ---
try:
    with open(pkl_path_mbart50, "rb") as f:
        log_history_mbart50 = pickle.load(f)
    with open(pkl_path_mbart25, "rb") as f:
        log_history_mbart25 = pickle.load(f)
    with open(pkl_path_mt5_base, "rb") as f:
        log_history_mt5_base = pickle.load(f)
    with open(pkl_path_M2M100_1_2B, "rb") as f:
        log_history_m2m100_1_2B = pickle.load(f)
    with open(pkl_path_M2M100_418M, "rb") as f:
        log_history_m2m100_418M = pickle.load(f)
except FileNotFoundError as e:
    print(f"错误：未能找到文件 {e}. 请检查路径是否正确。")
    exit()

# 2. 转换为 DataFrame
df_mbart50 = pd.DataFrame(log_history_mbart50)
df_mbart25 = pd.DataFrame(log_history_mbart25)
df_mt5_base = pd.DataFrame(log_history_mt5_base)
df_M2M100_1_2B = pd.DataFrame(log_history_m2m100_1_2B)
df_M2M100_418M = pd.DataFrame(log_history_m2m100_418M)

# 3. 准备数据字典
model_data = {
    "mBART-50": df_mbart50,
    "mBART-25": df_mbart25,
    "mT5-Base": df_mt5_base,
    "M2M100-1.2B": df_M2M100_1_2B,
    "M2M100-418M": df_M2M100_418M
}

# 颜色和标记配置，用于区分不同模型
MODEL_COLORS = {
    "mBART-50": "tab:blue",
    "mBART-25": "tab:orange",
    "mT5-Base": "tab:green",
    "M2M100-1.2B": "tab:red",
    "M2M100-418M": "tab:purple"
}
EVAL_MARKERS = {
    "mBART-50": "o",
    "mBART-25": "^",
    "mT5-Base": "s",
    "M2M100-1.2B": "D",
    "M2M100-418M": "X"
}

# --- 4. 数据预处理和缩放（集中处理所有模型） ---
max_steps = 0
processed_data = {}

for name, df in model_data.items():
    train_df = df.dropna(subset=['loss']).copy()
    eval_df = df.dropna(subset=['eval_loss']).copy()

    # 横轴缩放
    if not train_df.empty:
        train_df['scaled_step'] = train_df['step'] * train_batch_size
        train_df['smoothed_loss'] = train_df['loss'].ewm(span=SMOOTHING_WINDOW, adjust=False).mean()
        max_steps = max(max_steps, train_df['scaled_step'].max() if not train_df.empty else 0)

    if not eval_df.empty:
        eval_df['scaled_step'] = eval_df['step'] * train_batch_size

    processed_data[name] = {
        "train": train_df,
        "eval": eval_df
    }

print(
    f"原始最大训练步数: {max(model_data[name]['step'].max() for name, df in model_data.items() if not df.empty)}")
print(f"有效训练样本数 (Scaled Steps) 最大值: {max_steps:.0f}")


# ----------------------------------------------------
#               绘图并单独保存每个子图
# ----------------------------------------------------

# 定义一个辅助函数来创建并保存单个图
def plot_and_save_single_metric(metric_key, title, ylabel, legend_loc, save_suffix, is_loss=False):
    fig, ax = plt.subplots(figsize=(10, 6))  # 为每个图创建新的 Figure 和 Axes

    for model_name, data in processed_data.items():
        train_df = data['train']
        eval_df = data['eval']
        color = MODEL_COLORS[model_name]
        marker = EVAL_MARKERS[model_name]

        if is_loss:
            # 绘制平滑后的训练 Loss
            if not train_df.empty:
                ax.plot(train_df['scaled_step'], train_df['smoothed_loss'],
                        label=f'{model_name} Train Loss (SM={SMOOTHING_WINDOW})',
                        linestyle='-',
                        color=color,
                        alpha=0.6)
            # 绘制评估 Loss
            if not eval_df.empty:
                ax.plot(eval_df['scaled_step'], eval_df['eval_loss'],
                        label=f'{model_name} Eval Loss',
                        marker=marker,
                        markersize=5,
                        linestyle='--',
                        color=color,
                        alpha=0.9)
        else:
            # 绘制其他评估指标
            if not eval_df.empty and metric_key in eval_df.columns:
                ax.plot(eval_df['scaled_step'], eval_df[metric_key],
                        label=f'{model_name} {metric_key.replace("eval_", "").replace("_", " ").title()}',  # 美化标签
                        marker=marker,
                        markersize=5,
                        linestyle='-',
                        color=color,
                        linewidth=2)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(f'Effective Training Samples (Step $\\times$ Batch Size {train_batch_size})')
    ax.legend(loc=legend_loc, fontsize=8)
    ax.grid(True)  # 添加网格线

    # 设置 x 轴范围，将起始值设为 START_X_AXIS (-50)
    ax.set_xlim(START_X_AXIS, max_steps * 1.05)

    # 保存图片
    full_save_path = os.path.join(save_dir, f"{base_filename}_{save_suffix}.png")
    plt.tight_layout()
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭当前图，释放内存
    print(f"✅ 图表已成功保存到: {full_save_path}")


# 调用函数分别保存每个图
# 1. Loss 曲线
plot_and_save_single_metric(
    metric_key=None,
    title='Training and Evaluation Loss Comparison (Smoothed)',
    ylabel='Loss',
    legend_loc='upper right',
    save_suffix='loss_curves',
    is_loss=True
)

# 2. BLEU 指标曲线
plot_and_save_single_metric(
    metric_key='eval_bleu',
    title='Evaluation BLEU Score Comparison',
    ylabel='BLEU Score',
    legend_loc='lower right',
    save_suffix='bleu_score_curve'
)

# 3. RoUGE-L 指标曲线
plot_and_save_single_metric(
    metric_key='eval_rougeL',
    title='Evaluation ROUGE-L Score Comparison',
    ylabel='ROUGE-L Score',
    legend_loc='lower right',
    save_suffix='rougeL_score_curve'
)

# 4. Distinct-2 指标曲线
plot_and_save_single_metric(
    metric_key='eval_distinct-2',
    title='Evaluation Distinct-2 Score Comparison (for Diversity)',
    ylabel='Distinct-2 Score',
    legend_loc='upper right',
    save_suffix='distinct2_score_curve'
)

print(f"\n所有对比图已成功保存到目录: {save_dir}")