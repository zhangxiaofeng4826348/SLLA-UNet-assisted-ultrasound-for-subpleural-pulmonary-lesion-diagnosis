import pandas as pd
import joblib  # 1. 导入 joblib 替代 pickle
import os
import sys
import numpy as np # 导入 numpy 用于类型检查
# 保持为逻辑回归
from sklearn.linear_model import LogisticRegression

def main():
    print("--- [FILL] 开始推理 (LogisticRegression) ---")

    # 1. 动态设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 如果你的脚本是在 result 或其他子文件夹下运行，可能需要调整这里
    # 假设脚本就在项目根目录下，或者按你原有的逻辑保持不变
    
    # 定义文件路径
    model_path = os.path.join(current_dir, 'model', 'fill.pkl')
    data_path = os.path.join(current_dir, 'data', 'train', 'fill.csv')
    save_path = os.path.join(current_dir, 'result', 'fill_inference_result.csv')

    # 2. 加载模型 (核心修改部分)
    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型文件 {model_path}")
        return

    print(f"正在加载模型: {model_path}")
    
    try:
        # 使用 joblib 直接加载，无需 with open
        model = joblib.load(model_path)
        
        # --- 🛡️ 安全检查 (防止“数组”复活) ---
        print(f"👉 模型类型: {type(model)}")
        
        if isinstance(model, np.ndarray):
            raise ValueError(f"加载内容是一个 NumPy 数组 (Shape: {model.shape})，不是模型对象！请检查训练代码是否正确保存。")
        
        if not hasattr(model, "predict"):
            raise ValueError(f"加载的对象没有 predict 方法，可能不是有效的 scikit-learn 模型！")
            
        print("✅ 模型加载验证通过。")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 加载数据
    if not os.path.exists(data_path):
        print(f"❌ 错误：找不到数据文件 {data_path}")
        return

    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 4. 数据预处理
    cols_to_exclude = ['ID', 'label', 'Label', 'True_Label'] # 增加一些常见的非特征列
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]
    
    # 检查特征列是否为空
    if not feature_cols:
        print("❌ 错误: 未找到特征列，请检查 CSV 文件头。")
        return

    X = df[feature_cols]
    print(f"输入特征数量: {len(feature_cols)}")

    # 5. 模型推理
    print("正在计算...")
    try:
        y_pred = model.predict(X)
        
        # 逻辑回归通常支持 predict_proba
        if hasattr(model, "predict_proba"):
            # 获取属于类别 1 的概率
            y_prob = model.predict_proba(X)[:, 1]
        else:
            # 兼容性处理
            print("⚠️ 模型没有 predict_proba，尝试使用 decision_function")
            decision = model.decision_function(X)
            y_prob = 1 / (1 + np.exp(-decision))
            
    except Exception as e:
        print(f"❌ 推理发生错误: {e}")
        print("提示: 请检查用于推理的特征列顺序/数量是否与训练时一致。")
        return

    # 6. 保存结果
    # 确保结果目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    result_df = pd.DataFrame()
    if 'ID' in df.columns:
        result_df['ID'] = df['ID']
    
    # 兼容不同大小写的标签名
    if 'label' in df.columns:
        result_df['True_Label'] = df['label']
    elif 'Label' in df.columns:
        result_df['True_Label'] = df['Label']
        
    result_df['Pred_Label'] = y_pred
    result_df['Pred_Prob'] = y_prob

    try:
        result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ 结果已成功保存至: {save_path}")
    except Exception as e:
        print(f"❌ 保存结果文件失败 (可能是文件被打开了): {e}")

if __name__ == '__main__':
    main()