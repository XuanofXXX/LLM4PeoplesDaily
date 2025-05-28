import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# 1. 创建一个大型 NumPy 数据集
data_size = 1000000  # 数据样本数量
feature_dim = 2560     # 每个样本的特征维度
print(f"准备 NumPy 数据集，大小: ({data_size}, {feature_dim})...")
# 创建一个随机的 NumPy 数组作为我们的数据集
# 使用 float32 以模拟常见深度学习场景中的数据类型
numpy_data = np.random.randn(data_size, feature_dim).astype(np.float32)
numpy_labels = np.random.randint(0, 10, size=data_size).astype(np.int64) # 假设是10分类任务
print("NumPy 数据集准备完毕。\n")

# 2. 实现一个自定义 PyTorch Dataset
class MyCustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 注意：这里我们直接返回 NumPy 数组，DataLoader 会自动将其转换为 Tensor
        # 如果需要，也可以在此处转换为 Tensor:
        # return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)
        return self.data[idx], self.labels[idx]

print("创建自定义 Dataset 实例...")
custom_dataset = MyCustomDataset(numpy_data, numpy_labels)
print("Dataset 实例创建完毕。\n")

# --- 验证开始 ---

# 3. 直接遍历 Dataset
print("开始直接遍历 Dataset...")
start_time_dataset = time.time()
count_dataset = 0
for data, label in custom_dataset:
    # 模拟一些非常轻微的操作，实际应用中可能会有更复杂的操作
    # 这里我们只是简单地访问数据，以测量纯粹的迭代开销
    _ = data.shape
    _ = label.item()
    count_dataset += 1
end_time_dataset = time.time()
duration_dataset = end_time_dataset - start_time_dataset
print(f"直接遍历 Dataset 完成。")
print(f"遍历样本数: {count_dataset}")
print(f"耗时: {duration_dataset:.4f} 秒\n")

# 4. 通过 DataLoader 遍历 Dataset (单进程)
batch_size = 64
print(f"创建 DataLoader 实例 (batch_size={batch_size}, num_workers=0)...")
# num_workers=0 表示在主进程中加载数据
dataloader_single_worker = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print("DataLoader (单进程) 实例创建完毕。\n")

print("开始通过 DataLoader (单进程) 遍历...")
start_time_dataloader_single = time.time()
count_dataloader_single = 0
items_dataloader_single = 0
for batch_data, batch_labels in dataloader_single_worker:
    # batch_data 和 batch_labels 现在是 PyTorch Tensors
    # 模拟操作
    _ = batch_data.shape
    _ = batch_labels.sum() # 对批次标签做个简单操作
    count_dataloader_single += 1
    items_dataloader_single += len(batch_data)
end_time_dataloader_single = time.time()
duration_dataloader_single = end_time_dataloader_single - start_time_dataloader_single
print(f"通过 DataLoader (单进程) 遍历完成。")
print(f"遍历批次数: {count_dataloader_single}, 总样本数: {items_dataloader_single}")
print(f"耗时: {duration_dataloader_single:.4f} 秒\n")

# 5. 通过 DataLoader 遍历 Dataset (多进程)
# 注意：多进程在 Windows 上有时需要将 DataLoader 的创建和迭代放在 if __name__ == '__main__': 中。
# 在 Jupyter Notebook 或类似环境中，通常可以直接运行。
# num_workers 的选择通常取决于你的 CPU核心数。
num_workers_to_test = 4 # 你可以根据你的 CPU 调整这个值
print(f"创建 DataLoader 实例 (batch_size={batch_size}, num_workers={num_workers_to_test})...")
try:
    dataloader_multi_worker = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_to_test)
    print(f"DataLoader ({num_workers_to_test}个工作进程) 实例创建完毕。\n")

    print(f"开始通过 DataLoader ({num_workers_to_test}个工作进程) 遍历...")
    start_time_dataloader_multi = time.time()
    count_dataloader_multi = 0
    items_dataloader_multi = 0
    for batch_data, batch_labels in dataloader_multi_worker:
        _ = batch_data.shape
        _ = batch_labels.sum()
        count_dataloader_multi += 1
        items_dataloader_multi += len(batch_data)
    end_time_dataloader_multi = time.time()
    duration_dataloader_multi = end_time_dataloader_multi - start_time_dataloader_multi
    print(f"通过 DataLoader ({num_workers_to_test}个工作进程) 遍历完成。")
    print(f"遍历批次数: {count_dataloader_multi}, 总样本数: {items_dataloader_multi}")
    print(f"耗时: {duration_dataloader_multi:.4f} 秒\n")

except RuntimeError as e:
    if " περισσότεprocesses" in str(e) or "BrokenPipeError" in str(e) or "can't start new thread" in str(e) or "unable to start new thread" in str(e):
        print(f"注意: 在某些环境 (特别是 Windows 或资源受限的 Notebook) 中直接运行多进程 DataLoader 可能会遇到问题。")
        print(f"错误信息: {e}")
        print("如果遇到此问题，请尝试将 DataLoader 的创建和迭代部分放在 `if __name__ == '__main__':` 块中运行脚本，或者减少 `num_workers`。")
        duration_dataloader_multi = float('inf') # 标记为失败或不可用
    else:
        raise e


# --- 总结 ---
print("="*30)
print("实验结果总结:")
print("="*30)
print(f"直接遍历 Dataset ({count_dataset} 项): {duration_dataset:.4f} 秒")
print(f"通过 DataLoader (num_workers=0, {items_dataloader_single} 项): {duration_dataloader_single:.4f} 秒")
if duration_dataloader_multi != float('inf'):
    print(f"通过 DataLoader (num_workers={num_workers_to_test}, {items_dataloader_multi} 项): {duration_dataloader_multi:.4f} 秒")
    if duration_dataloader_single > duration_dataset:
        print(f"\n➡️ DataLoader (num_workers=0) 比直接遍历 Dataset 慢了约 {(duration_dataloader_single / duration_dataset - 1) * 100:.2f}%")
    if duration_dataloader_multi > duration_dataset:
        print(f"➡️ DataLoader (num_workers={num_workers_to_test}) 比直接遍历 Dataset 慢了约 {(duration_dataloader_multi / duration_dataset - 1) * 100:.2f}% (但可能因为并行化而比 num_workers=0 快)")
    elif duration_dataloader_multi < duration_dataloader_single :
         print(f"\n✅ DataLoader (num_workers={num_workers_to_test}) 比 DataLoader (num_workers=0) 快了约 {(1 - duration_dataloader_multi / duration_dataloader_single) * 100:.2f}%，显示了多进程的优势。")
    if duration_dataloader_multi < duration_dataset:
        print(f"🔥 注意：在某些情况下，如果数据预处理非常简单且 num_workers 合理，多进程 DataLoader 甚至可能比直接串行遍历 Dataset 更快，但这不常见于纯迭代 NumPy 数组的场景。")

else:
    print(f"多进程 DataLoader (num_workers={num_workers_to_test}) 测试未能成功完成。")
    if duration_dataloader_single > duration_dataset:
        print(f"\n➡️ DataLoader (num_workers=0) 比直接遍历 Dataset 慢了约 {(duration_dataloader_single / duration_dataset - 1) * 100:.2f}%")

print("\n实验分析:")
print("1. **直接遍历 Dataset** 通常是最快的，因为它只涉及到 Python 的迭代器协议和 NumPy 数组的直接访问。")
print("2. **DataLoader (num_workers=0)** 会引入额外的开销，因为数据需要经过 `collate_fn` (即使是默认的) 进行批处理，并且从 NumPy 数组转换为 PyTorch Tensor (如果 `__getitem__` 返回的是 NumPy 数组)。这些操作在主进程中串行执行。")
print("3. **DataLoader (num_workers > 0)** 会使用多个子进程来并行加载和预处理数据。虽然这可以显著加速整个数据加载流程（特别是当 `__getitem__` 中的预处理比较耗时时），但进程的创建、管理、数据在进程间的传递（序列化和反序列化）本身也会引入开销。对于从内存中的 NumPy 数组读取这样IO开销极小、CPU计算也极少的情况，多进程的开销可能会超过其带来的并行优势，导致它比 `num_workers=0` 甚至直接遍历 `Dataset` 更慢。")
print("   - 然而，如果 `__getitem__` 中包含复杂的转换或从磁盘读取数据，`num_workers > 0` 通常会带来显著的性能提升。")
print("   - `DataLoader` 的主要优势在于它能够隐藏数据加载的延迟，使得 GPU 在训练时不会因为等待数据而空闲。")