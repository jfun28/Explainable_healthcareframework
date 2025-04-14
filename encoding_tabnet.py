import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# def train_tabnet_classifier(X_train, y_train, X_valid, y_valid):
#     """TabNet 분류기 학습 함수"""
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     tabnet_params = {
#         "n_d": 8,
#         "n_a": 8,
#         "n_steps": 5,
#         "gamma": 1.3,
#         "cat_idxs": [],
#         "cat_dims": [],
#         "cat_emb_dim": [],
#         "n_independent": 2,
#         "n_shared": 2,
#         "epsilon": 1e-15,
#         "momentum": 0.02,
#         "lambda_sparse": 0.001,
#         "seed": 0,
#         "clip_value": 1,
#         "verbose": 1,
#         "optimizer_fn": torch.optim.Adam,
#         "optimizer_params": {'lr': 0.02},
#         "scheduler_fn": None,
#         "scheduler_params": {},
#         "mask_type": 'sparsemax',
#         "input_dim": X_train.shape[1],
#         "output_dim": len(np.unique(y_train)),
#         "device_name": 'auto',
#         "n_shared_decoder": 1,
#         "n_indep_decoder": 1,
#         "grouped_features": []
#     }
    
#     tabnet_clf = TabNetClassifier(**tabnet_params)
#     max_epochs = 300
    
#     tabnet_clf.fit(
#         X_train=X_train, 
#         y_train=y_train,
#         eval_set=[(X_valid, y_valid)],
#         max_epochs=max_epochs,
#         patience=50,
#         batch_size=5000,
#         virtual_batch_size=5000,
#         num_workers=1,
#         drop_last=False,
#     )
    
#     return tabnet_clf

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
    max_epochs = 300
    
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
    
    
    print(f"원본 특성 shape: {X.shape}")
    print(f"샘플별 마스크 평균 shape: {masks.shape}")
   
    augmented_features = np.hstack([X, masks])
    augmented_features2 = X+masks

    return masks


def create_augmented_features2(tabnet_model, X):
    masks, _=tabnet_model.explain(X)
    predict_proba=tabnet_model.predict_proba(X)
    
    print(f"원본 특성 shape: {X.shape}")
    print(f"샘플별 마스크 평균 shape: {masks.shape}")
   
    augmented_features = np.hstack([X, masks,predict_proba])

    return augmented_features







































# # encoding_tabnet.py

# '''
# 이 모듈은 Tabnet기반 encoding으로 tabnet에서 학습과정을 통해 mask 값을 통해 변수들을 embeddings을 한다.
# 그렇게 encoding한 임베딩 값들이 추후에 xgboost input으로 들어가게 된다.
# '''

# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from pytorch_tabnet.tab_model import TabNetClassifier
# import numpy as np

# def train_tabnet_classifier(X_train, y_train, X_valid, y_valid):
#     """TabNet 분류기 학습 함수"""
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     tabnet_params = {
#         "n_d": 8,
#         "n_a": 8,
#         "n_steps": 5,
#         "gamma": 1.3,
#         "cat_idxs": [],
#         "cat_dims": [],
#         "cat_emb_dim": [],
#         "n_independent": 2,
#         "n_shared": 2,
#         "epsilon": 1e-15,
#         "momentum": 0.02,
#         "lambda_sparse": 0.001,
#         "seed": 0,
#         "clip_value": 1,
#         "verbose": 1,
#         "optimizer_fn": torch.optim.Adam,
#         "optimizer_params": {'lr': 0.02},
#         "scheduler_fn": None,
#         "scheduler_params": {},
#         "mask_type": 'sparsemax',
#         "input_dim": 6,
#         "output_dim": [3],
#         "device_name": 'auto',
#         "n_shared_decoder": 1,
#         "n_indep_decoder": 1,
#         "grouped_features": []
#     }
    
#     tabnet_clf = TabNetClassifier(**tabnet_params)
#     max_epochs = 100
    
#     tabnet_clf.fit(
#         X_train=X_train, 
#         y_train=y_train,
#         eval_set=[(X_valid, y_valid)],
#         max_epochs=max_epochs,
#         patience=100,
#         batch_size=5000,
#         virtual_batch_size=5000,
#         num_workers=1,
#         drop_last=False,
#     )
    
#     return tabnet_clf

# def extract_tabnet_embeddings(tabnet_model, X_tensor, device=None, batch_size=5000):
#     """TabNet 모델에서 임베딩 추출 함수"""
#     if device is None:
#         device = next(tabnet_model.network.parameters()).device
    
#     tabnet_model.network.eval()
#     tabnet_model.network.to(device)
#     X_tensor = X_tensor.to(device)
    
#     dataset = TensorDataset(X_tensor)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
#     all_embeddings = []
    
#     with torch.no_grad():
#         for (xb,) in loader:
#             embeddings, _ = tabnet_model.network.forward_masks(xb)
                    
#             if isinstance(embeddings, dict):
#                 if 'embeddings' in embeddings:
#                     final_emb = embeddings['embeddings'][-1]
#                 else:
#                     final_emb = list(embeddings.values())[-1]
#             else:
#                 # embeddings가 리스트인 경우, 마지막 decision step의 embedding 사용
#                 if isinstance(embeddings, list):
#                     final_emb = embeddings[-1]
#                 else:
#                     final_emb = embeddings
            
#             # final_emb의 shape 확인
#             print("Shape of final_emb:", final_emb.shape)
            
#             all_embeddings.append(final_emb.cpu().numpy())
    
#     final_embeddings = np.concatenate(all_embeddings, axis=0) # 부분에서 그 리스트에 있는 모든 배치 임베딩을 axis=0 (첫 번째 차원)을 따라 합칩니다.
#     # 최종 shape 출력
#     print("Final embeddings shape:", final_embeddings.shape)
#     return final_embeddings