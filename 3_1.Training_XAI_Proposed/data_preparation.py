# data_preparation.py
'''
각 generation에서 생성하고 train_valid_test으로 분리한 데이터세트를 불러온다.  
'''
import numpy as np
import torch
import os
from sklearn.preprocessing import LabelEncoder
current_dir = os.path.dirname(os.path.abspath(__file__))
    
# 프로젝트 루트 디렉토리 경로 (현재 파일의 상위 디렉토리)
path = os.path.dirname(current_dir)

def preprocess_data(generation):
    """데이터 전처리 함수"""
   # 학습 데이터 불러오기
    print("경로 확인",path)
    X_train = np.loadtxt(path+f'\\Synthetic_data\\train_valid_test\\서울초_{generation}_X_train.txt', delimiter=',')
    y_train = np.loadtxt(path+f'\\Synthetic_data\\train_valid_test\\서울초_{generation}_y_train.txt', delimiter=',')

    # 검증 데이터 불러오기
    X_valid = np.loadtxt(path+f'\\Synthetic_data\\train_valid_test\\서울초_{generation}_X_valid.txt', delimiter=',')
    y_valid = np.loadtxt(path+f'\\Synthetic_data\\train_valid_test\\서울초_{generation}_y_valid.txt', delimiter=',')

    # 테스트 데이터 불러오기
    X_test = np.loadtxt(path+f'\\Synthetic_data\\train_valid_test\\서울초_{generation}_X_test.txt', delimiter=',')
    y_test = np.loadtxt(path+f'\\Synthetic_data\\train_valid_test\\서울초_{generation}_y_test.txt', delimiter=',')

    
    # # 트레이닝 데이터에 맞춰 인코더 학습
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_valid = encoder.transform(y_valid)
    y_test = encoder.transform(y_test)
        
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def convert_to_tensor(X_train,X_valid, X_test):
    """데이터를 PyTorch 텐서로 변환"""
    X_train_torch = torch.from_numpy(X_train.astype(np.float32))
    X_valid_torch = torch.from_numpy(X_test.astype(np.float32))
    X_test_torch = torch.from_numpy(X_test.astype(np.float32))
    return X_train_torch, X_valid_torch, X_test_torch