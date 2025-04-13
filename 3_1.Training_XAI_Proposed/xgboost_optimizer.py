# import optuna
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score
# import numpy as np
# from sklearn.utils.class_weight import compute_class_weight

# def objective_xgboost(trial, X_train, X_test, y_train, y_test):
#     """XGBoost 하이퍼파라미터 최적화를 위한 목적 함수"""
    
#     # 클래스 가중치 계산
#     class_weights = compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y_train),
#         y=y_train
#     )
#     class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
#     # 샘플 가중치 생성
#     sample_weights = np.array([class_weight_dict[y] for y in y_train])
    
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
#         'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
#         'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
#         'random_state': 42,
#         'objective': 'multi:softmax',
#         'num_class': 3,
#         'use_label_encoder': False,
#         'eval_metric': 'mlogloss'
#     }
    
#     model = XGBClassifier(**params)
#     model.fit(X_train, y_train, sample_weight=sample_weights)
#     y_pred = model.predict(X_test)
#     return accuracy_score(y_test, y_pred)

# def optimize_and_compare_models(X_train, X_test, y_train, y_test, n_trials=10, return_proba=False):
#     """XGBoost 모델 최적화 및 비교 함수"""
#     results = {}
    
#     # 클래스 가중치 계산
#     class_weights = compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y_train),
#         y=y_train
#     )
#     class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
#     # 샘플 가중치 생성
#     sample_weights = np.array([class_weight_dict[y] for y in y_train])
    
#     # XGBoost 최적화
#     study_xgb = optuna.create_study(direction='maximize')
#     study_xgb.optimize(lambda trial: objective_xgboost(trial, X_train, X_test, y_train, y_test), 
#                     n_trials=n_trials)
    
#     # 최적의 모델 생성 및 결과 저장
#     best_params = study_xgb.best_params
#     best_xgb = XGBClassifier(**best_params, random_state=42)
#     best_xgb.fit(X_train, y_train, sample_weight=sample_weights)
    
#     xgb_pred = best_xgb.predict(X_test)
#     results['XGBoost'] = {
#         'accuracy': accuracy_score(y_test, xgb_pred),
#         'best_params': study_xgb.best_params,
#         'class_weights': class_weight_dict
#     }
    
#     if return_proba:
#         xgb_pred_proba=best_xgb.predict_proba(X_test)
#         return results, xgb_pred, best_xgb, xgb_pred_proba
#     else:
#         return results, xgb_pred, best_xgb

# xgboost_optimizer.py
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

'''
optuna 라이브러리를 사용하여 베이지안 최적화를 수행하였고 
가장 최적의 xgboost를 best_xgb로 최종적으로 사용하였다.
'''
def objective_xgboost(trial, X_train, X_test, y_train, y_test):
    """XGBoost 하이퍼파라미터 최적화를 위한 목적 함수"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'random_state': 42,
        'objective': 'multi:softmax',
        'num_class': 3,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def optimize_and_compare_models(X_train, X_test, y_train, y_test, n_trials=10,return_proba=False):
    """XGBoost 모델 최적화 및 비교 함수"""
    results = {}
    
    # XGBoost 최적화
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(lambda trial: objective_xgboost(trial, X_train, X_test, y_train, y_test), 
                    n_trials=n_trials)
    
    # 최적의 모델 생성 및 결과 저장
    best_xgb = XGBClassifier(**study_xgb.best_params, random_state=42)
    best_xgb.fit(X_train, y_train)
    xgb_pred = best_xgb.predict(X_test)
    results['XGBoost'] = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'best_params': study_xgb.best_params
    }
    
    if return_proba:
        xgb_pred_proba=best_xgb.predict_proba(X_test)
        return results, xgb_pred, best_xgb, xgb_pred_proba
    else:
        return results, xgb_pred, best_xgb