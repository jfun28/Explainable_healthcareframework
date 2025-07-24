import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


from sklearn.utils.class_weight import compute_class_weight

def train_tabnet_classifier(X_train, y_train, X_valid, y_valid):
    """TabNet 분류기 학습 함수 - 샘플별 가중치 적용"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 클래스 분포에 따른 자동 가중치 계산
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    
    # 가중치 확인을 위한 출력
    print("자동 계산된 클래스 가중치:", class_weights)
    
    # 클래스별 가중치를 샘플별 가중치로 변환
    sample_weights = np.ones(len(y_train))
    for i, y in enumerate(y_train):
        for j, cls in enumerate(unique_classes):
            if y == cls:
                sample_weights[i] = class_weights[j]
                break
    
    # 가중치 텐서 생성
    weights_tensor = torch.FloatTensor(sample_weights)
    
    tabnet_params = {
        "n_d": 8,
        "n_a": 8,
        "n_steps": 5,
        "gamma": 1.3,
        "cat_idxs": [],
        "cat_dims": [],
        "cat_emb_dim": [],
        "n_independent": 2,
        "n_shared": 2,
        "epsilon": 1e-15,
        "momentum": 0.02,
        "lambda_sparse": 0.001,
        "seed": 0,
        "clip_value": 1,
        "verbose": 1,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": {'lr': 0.02},
        "scheduler_fn": None,
        "scheduler_params": {},
        "mask_type": 'sparsemax',
        "input_dim": X_train.shape[1],
        "output_dim": len(unique_classes),
        "device_name": 'auto',
        "n_shared_decoder": 1,
        "n_indep_decoder": 1,
        "grouped_features": []
    }
    
    tabnet_clf = TabNetClassifier(**tabnet_params)
    max_epochs = 500
    
    tabnet_clf.fit(
        X_train=X_train, 
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        max_epochs=max_epochs,
        patience=50,
        batch_size=5000,
        virtual_batch_size=5000,
        num_workers=1,
        drop_last=False,
        weights=weights_tensor  # 샘플별 가중치 전달
    )
    
    return tabnet_clf



def create_augmented_features(tabnet_model, X):
    masks, _=tabnet_model.explain(X)
    return masks



