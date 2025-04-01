# utils.py
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

def standardize_features(X_train_features, X_test_features):
    """특성 표준화 함수"""
    standard_scaler = StandardScaler()
    X_train_std = standard_scaler.fit_transform(X_train_features)
    X_test_std = standard_scaler.transform(X_test_features)
    return X_train_std, X_test_std, standard_scaler

def calculate_metrics(y_test, y_pred):
    """성능 지표 계산 함수"""
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    return accuracy, f1, precision, recall

def save_model(model, file_path):
    """모델 저장 함수"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def print_results(results):
    """모델 결과 출력 함수"""
    for model_name, model_results in results.items():
        print(f"\n{model_name} Results:")
        print(f"Best Accuracy: {model_results['accuracy']:.4f}")
        print("Best Parameters:")
        for param, value in model_results['best_params'].items():
            print(f"  {param}: {value}")