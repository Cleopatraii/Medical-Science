import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 加载训练集和测试集
    train_df = pd.read_csv("train_ml.csv")
    test_df = pd.read_csv("test_ml.csv")

    # 定义特征列和标签
    feature_cols = [
        'GCS', 'Heart Rate', 'Systolic BP', 'ShockIndex_current',
        'Albumin', 'WBC', 'CRP', 'Body Temp', 'ShockIndex_trend_4h'
    ]
    label_col = 'shock_index_critical_12h'

    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values

    # === 基线模型 ===
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    y_prob_base = baseline.predict_proba(X_test)[:, 1]  # 概率分数

    # === 随机森林模型 ===
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred_rf = model.predict(X_test)
    y_prob_rf = model.predict_proba(X_test)[:, 1]  # 概率分数

    print("Baseline Accuracy:", accuracy_score(y_test, y_pred_base))
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

    # 画混淆矩阵
    def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar()
        plt.show()

    plot_confusion(y_test, y_pred_base, title="Baseline Confusion Matrix")
    plot_confusion(y_test, y_pred_rf, title="Random Forest Confusion Matrix")

    # ROC & PR 曲线
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    auc_base = auc(fpr_base, tpr_base)
    auc_rf = auc(fpr_rf, tpr_rf)

    precision_base, recall_base, _ = precision_recall_curve(y_test, y_prob_base)
    ap_base = average_precision_score(y_test, y_prob_base)
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)
    ap_rf = average_precision_score(y_test, y_prob_rf)

    plt.figure()
    plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC={auc_base:.2f})')
    plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC={auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(recall_base, precision_base, label=f'Baseline (AP={ap_base:.2f})')
    plt.plot(recall_rf, precision_rf, label=f'RandomForest (AP={ap_rf:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
