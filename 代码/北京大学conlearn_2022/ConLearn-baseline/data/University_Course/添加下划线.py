import pandas as pd


def replace_spaces_with_underscores(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 获取所有列名
    columns = data.columns

    # 检查列A和列B是否在数据中
    if 'A' in columns and 'B' in columns:
        # 替换列A和列B中的空格为下划线
        data['A'] = data['A'].str.replace(' ', '_')
        data['B'] = data['B'].str.replace(' ', '_')

        # 保存修改后的DataFrame到CSV
        output_path = file_path.replace('.csv', '.csv')
        data.to_csv(output_path, index=False)

        return output_path
    else:
        raise KeyError("Columns 'A' or 'B' were not found in the data.")


# 用您的文件路径替换以下路径
file_path = 'val_concept_pair.csv'
modified_file_path = replace_spaces_with_underscores(file_path)
print(f"Modified file saved to: {modified_file_path}")
