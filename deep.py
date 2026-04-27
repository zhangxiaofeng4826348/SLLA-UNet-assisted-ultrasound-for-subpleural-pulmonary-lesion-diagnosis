import pandas as pd
import pickle
import os
import sys
# 修改导入为梯度提升树
from sklearn.ensemble import GradientBoostingClassifier

def main():
    print("--- [DEEP] 开始推理 (GradientBoosting) ---")

    # 1. 动态设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 定义文件路径
    model_path = os.path.join(current_dir, 'model', 'deep.pkl')
    data_path = os.path.join(current_dir, 'data', 'train', 'deep.csv')
    save_path = os.path.join(current_dir,'result', 'deep_inference_result.csv')

    # 2. 加载模型
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    print(f"加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"模型类型: {type(model)}")

    # 3. 加载数据
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}")
        return

    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 4. 数据预处理
    cols_to_exclude = ['ID', 'label', 'Label']
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]
    X = df[feature_cols]
    print(f"输入特征数量: {len(feature_cols)}")

    # 5. 模型推理
    print("正在计算...")
    try:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"推理发生错误: {e}")
        return

    # 6. 保存结果
    result_df = pd.DataFrame()
    if 'ID' in df.columns:
        result_df['ID'] = df['ID']
    if 'label' in df.columns:
        result_df['True_Label'] = df['label']
        
    result_df['Pred_Label'] = y_pred
    result_df['Pred_Prob'] = y_prob

    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至: {save_path}")

if __name__ == '__main__':
    main()