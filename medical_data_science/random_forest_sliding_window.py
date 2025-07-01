import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # 加载数据
    df = pd.read_csv("train_ml.csv")

    # 标记 patient_id
    df['patient_id'] = (df['hours_since_admission'] == 0).cumsum()

    # 特征与标签
    feature_cols = [
        'GCS', 'Heart Rate', 'Systolic BP', 'ShockIndex_current',
        'Albumin', 'WBC', 'CRP', 'Body Temp', 'ShockIndex_trend_4h'
    ]
    label_col = 'shock_index_critical_12h'


    # 滑动窗口构造函数
    def build_sliding_windows(df, window_size):
        X, y = [], []
        for _, group in df.groupby("patient_id"):
            group = group.sort_values("hours_since_admission")
            features = group[feature_cols].values
            labels = group[label_col].values
            if len(group) < window_size:
                continue
            for i in range(len(group) - window_size + 1):
                window = features[i:i + window_size]
                stats = []
                stats.append(np.mean(window, axis=0))
                stats.append(np.std(window, axis=0))
                stats.append(np.min(window, axis=0))
                stats.append(np.max(window, axis=0))
                X.append(np.concatenate(stats))  # 拼接成一维特征向量
                y.append(labels[i + window_size - 1])
        return np.array(X), np.array(y)


    # 构建窗口数据
    window_size = 6
    X, y = build_sliding_windows(df, window_size)

    # 训练与测试划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 随机森林模型
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 结果输出
    print("Sliding Window RandomForest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


    def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = range(len(set(y_true)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)

        plt.xlabel('Predicted')
        plt.ylabel('True')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.show()


    # 训练和预测
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 打印评估结果
    print("Sliding Window RandomForest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 画混淆矩阵（与Baseline风格一致）
    plot_confusion(y_test, y_pred, title="Sliding Window Random Forest Confusion Matrix")

    # ROC曲线与AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:  # 有的模型没有proba，比如SVM
        y_score = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()

    # Precision-Recall曲线
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.show()

    base_features = ['GCS', 'Heart Rate', 'Systolic BP', 'ShockIndex_current',
                     'Albumin', 'WBC', 'CRP', 'Body Temp', 'ShockIndex_trend_4h']

    # 拼接成 mean, std, min, max
    feature_names = []
    for stat in ['mean', 'std', 'min', 'max']:
        feature_names.extend([f'{feat}_{stat}' for feat in base_features])

    print(feature_names)

    importances = model.feature_importances_

    feat_importances = pd.Series(importances, index=feature_names)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

    print(feat_importances.sort_values(ascending=False))

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_score),
        "Average Precision (AP)": average_precision_score(y_test, y_score)
    }
    print(pd.DataFrame([results]))