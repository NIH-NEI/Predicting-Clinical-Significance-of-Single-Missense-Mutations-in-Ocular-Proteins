import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

def bootstrap_auc(y_true, y_pred, n_bootstrap=1000):
    bootstrapped_aucs = []
    n_size = len(y_true)

    for i in range(n_bootstrap):
        indices = resample(np.arange(n_size), replace=True, n_samples=n_size, random_state=i)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_aucs.append(score)

    return np.array(bootstrapped_aucs)

# Calculate bootstrapped AUCs for each model
n_bootstrap = 1000
aucs = {}

model_predictions = {
    'Decision Tree': y_pred_dt,
    'Random Forest': y_pred_rf,
    'XGBoost': y_pred_xgb,
    'SVM': y_pred_svm
}

for model_name, y_pred in model_predictions.items():
    aucs[model_name] = bootstrap_auc(y_test, y_pred, n_bootstrap=n_bootstrap)

# Calculate p-values
def calculate_p_value(auc1, auc2):
    diff = auc1 - auc2
    p_value = np.sum(diff < 0) / len(diff)
    return p_value

p_values = {}

model_names = list(model_predictions.keys())
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        model1 = model_names[i]
        model2 = model_names[j]
        p_values[f'{model1} vs {model2}'] = calculate_p_value(aucs[model1], aucs[model2])

for comparison, p_value in p_values.items():
    print(f"{comparison} p-value: {p_value:.4f}")
