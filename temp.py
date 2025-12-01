import torch
from torch.utils.data import TensorDataset
import os
import shutil

# --- 配置路径 ---
# 请确保这里指向你生成错误的那个 .pt 文件路径
FILE_PATH = "cache/lambda_icl_qwen_0.6b/pretrain_datasetv2.pt" 
BACKUP_PATH = FILE_PATH + ".bak"

def fix_dataset_indices(file_path):
    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        return

    print(f"正在加载数据集: {file_path} ...")
    # 加载数据集 (weights_only=False 是为了兼容旧版 pytorch 保存习惯，如果是新版可以直接去掉)
    dataset = torch.load(file_path, weights_only=False)
    
    # 获取内部张量
    # 根据之前的代码，顺序是: (query_embs, target_dists, value_targets, actor_masks)
    tensors = dataset.tensors
    if len(tensors) != 4:
        print(f"警告：数据集张量数量不符合预期 (期待4个，实际{len(tensors)}个)。")
        # 如果你用的是旧版代码生成的只有2个张量的版本，请停止操作
        return

    query_embs, old_dists, value_targets, actor_masks = tensors
    
    print("正在修复索引偏移...")
    
    # 创建一个新的全0分布容器
    new_dists = torch.zeros_like(old_dists)
    
    # --- 搬运数据 ---
    # 1. 没有问题的列，直接复制
    # Lambda 0.5 (Index 10)
    new_dists[:, 10] = old_dists[:, 10]
    # Lambda 0.8 (Index 16)
    new_dists[:, 16] = old_dists[:, 16]
    # Lambda 0.9 (Index 18)
    new_dists[:, 18] = old_dists[:, 18]
    # Lambda 1.0 (Index 20)
    new_dists[:, 20] = old_dists[:, 20]
    
    # 2. 有问题的列，移动位置
    # Lambda 0.6: Index 11 (错) -> Index 12 (对)
    if old_dists[:, 11].sum() > 0:
        print(f"  - 发现 Index 11 (Lambda 0.6) 的数据，正在移动到 Index 12...")
        new_dists[:, 12] = old_dists[:, 11]
        
    # Lambda 0.7: Index 13 (错) -> Index 14 (对)
    if old_dists[:, 13].sum() > 0:
        print(f"  - 发现 Index 13 (Lambda 0.7) 的数据，正在移动到 Index 14...")
        new_dists[:, 14] = old_dists[:, 13]

    # --- 验证完整性 ---
    # 理论上总概率质量应该守恒 (误差允许范围内)
    diff = (new_dists.sum() - old_dists.sum()).abs().item()
    if diff > 1e-4:
        print(f"错误：数据搬运后总概率质量发生了变化 (Diff: {diff})！请检查逻辑。")
        return
    else:
        print("完整性验证通过。")

    # --- 备份与保存 ---
    print(f"备份原文件到: {BACKUP_PATH}")
    shutil.copy(file_path, BACKUP_PATH)
    
    print("正在保存修复后的数据集...")
    new_dataset = TensorDataset(query_embs, new_dists, value_targets, actor_masks)
    torch.save(new_dataset, file_path)
    print(f"成功！已覆盖文件: {file_path}")

if __name__ == "__main__":
    fix_dataset_indices(FILE_PATH)