import pandas as pd
from sklearn.model_selection import train_test_split

# 划分数据集
def split_dataset(filepath, save_dir, test_size=0.1, val_size=0.1, random_state=42):
    """
    加载数据并将其分割为训练集、验证集和测试集。

    参数:
    - filepath: 数据集的文件路径。
    - test_size: 测试集占总数据集的比例。
    - val_size: 验证集占（总数据集 - 测试集）的比例。

    返回:
    - train_data: 训练集。
    - val_data: 验证集。
    - test_data: 测试集。
    """
    # 加载数据
    data = pd.read_csv(filepath)

    # 先分割出测试集
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    # 再从剩余数据中分割出验证集，注意调整val_size以保证验证集是剩余数据的10%
    adjusted_val_size = val_size / (1.0 - test_size)
    train_data, val_data = train_test_split(train_val_data, test_size=adjusted_val_size, random_state=random_state)

    # 确保保存目录存在
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存数据集到指定路径
    train_data.to_csv(f"{save_dir}/train_data.csv", index=False)
    val_data.to_csv(f"{save_dir}/val_data.csv", index=False)
    test_data.to_csv(f"{save_dir}/test_data.csv", index=False)


# 通过训练集获取边数据集
def process_and_save_edges(train_data_path, node_path, edges_output_path, use_real_label):
    """
    处理几何数据和节点数据，提取边信息，并将结果保存到CSV文件。

    参数:
    - train_data_path: 训练集概念对CSV文件的路径。
    - node_path: 节点数据CSV文件的路径id+name+lm+node。主要使用node_emb。
    - edges_output_path: 保存提取的边数据CSV文件的路径。
    - use_real_label: 布尔值，决定用弱标签还是真实标签。use_real_label=true则使用真实标签。
    """
    # 加载数据集
    train_data_df = pd.read_csv(train_data_path)
    node_df = pd.read_csv(node_path)

    # 创建从节点名称到节点ID的映射
    name_to_id_map = node_df.set_index('node_name')['node_id'].to_dict()

    # 初始化一个列表来存储边
    edges = []

    # 遍历几何数据dataframe
    for index, row in train_data_df.iterrows():
        if (use_real_label and row['result'] > 0) or (not use_real_label and row['RefD'] > 0):
            # 获取源节点和目标节点的名称
            source_node_name = row['A']
            target_node_name = row['B']

            # 使用映射将节点名称转换为节点ID
            source_node_id = name_to_id_map.get(source_node_name, None)
            target_node_id = name_to_id_map.get(target_node_name, None)

            # 如果找到了两个ID，则将该边添加到列表中
            if source_node_id is not None and target_node_id is not None:
                edges.append((source_node_id, target_node_id))

    # 将边列表转换为DataFrame
    edges_df = pd.DataFrame(edges, columns=['source_node_id', 'target_node_id'])

    # 将边数据dataframe保存到新的CSV文件
    edges_df.to_csv(edges_output_path, index=False)

if __name__ == "__main__":
    # 划分数据集       （加载路径 保存路径）
    # split_dataset('../data/lectureBankData/cs_concept_pairs.csv', '../Dataset/lectureBankData')

    process_and_save_edges('../data/MOOC/train_concept_pair.csv',
                           '../data/MOOC/MOOC(id+name+lm+node).csv',
                           '../data/MOOC/edges_dataset.csv', True)





