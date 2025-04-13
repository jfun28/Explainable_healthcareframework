# main.py
import pandas as pd
import torch
import numpy as np
# 로컬 모듈 임포트
from data_preparation import preprocess_data, convert_to_tensor
from encoding_tabnet import train_tabnet_classifier, create_augmented_features2,create_augmented_features
from xgboost_optimizer import optimize_and_compare_models
from utils import standardize_features, calculate_metrics, save_model, print_results
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 디렉토리 경로 (현재 파일의 상위 디렉토리)
path = os.path.dirname(current_dir)
def main():
    """메인 함수"""
    # 빈 리스트 생성하여 결과 저장
    model_results_list = []
    
    # Generation_list는 외부에서 정의되어야 함
    Generation_list = ['smote','adasyn','copulagan','ctgan','nbsynthetic']
    
    for generation in Generation_list:
        print(f"\n처리 중인 생성 모델: {generation}")
        
        # 1. 데이터 전처리
        X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(generation)
        
        print("y_train",np.unique(y_train))
        # 2. TabNet 모델 학습
        tabnet_model = train_tabnet_classifier(X_train, y_train, X_valid, y_valid)

        # 3. 모델 저장
        model_save_path = path+f'\\3_1.Training_XAI_Proposed\\model_hist\\{generation}_proposed_tabnetEmbedd2.pickle'
        save_model(tabnet_model, model_save_path)
        tabnet_probs = tabnet_model.predict_proba(X_test)
        
        # 4. TabNet 확률 예측
        
        # 5. TabNet 임베딩 추출
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_features = create_augmented_features(tabnet_model, X_train)
        test_features = create_augmented_features(tabnet_model, X_test)
    
        
        # 6. XGBoost 모델 최적화 및 학습
        results, xgb_pred, best_xgb, xgb_probs = optimize_and_compare_models(
            train_features, 
            test_features, 
            y_train.astype(int), 
            y_test.astype(int), 
            n_trials=10,
            return_proba=True  # 확률값 반환하도록 수정
        )
        
        # 7. TabNet과 XGBoost 확률 합산   
        print("tabnet_probs",tabnet_probs)
        print("xgb_probs",xgb_probs)

        # 확률 합산
        
        combined_probs = (tabnet_probs + xgb_probs)
        
        # # 가장 높은 확률을 가진 클래스 선택
        ensemble_pred = np.argmax(combined_probs, axis=1)
        # 8. 결과 출력
        print_results(results)
        
        # 9. XGBoost 모델 저장
        xgb_save_path = path+f'\\3_1.Training_XAI_Proposed\\model_hist\\{generation}_proposed.pickle'
        save_model(best_xgb, xgb_save_path)
        
        # 10. 성능 지표 계산 (앙상블 예측 사용)
        accuracy, f1, precision, recall = calculate_metrics(y_test, ensemble_pred)
        
        # 11. 결과를 딕셔너리로 저장
        result_dict = {
            'Model': f"{generation}-proposed",
            'Accuracy': round(accuracy, 5),
            'F1score': round(f1, 5),
            'Precision': round(precision, 5),
            'Recall': round(recall, 5)
        }
        
        # 12. 리스트에 딕셔너리 추가
        model_results_list.append(result_dict)
    
    # 13. 데이터프레임 생성
    df_proposed = pd.DataFrame(model_results_list)
    print("\n결과 요약:")
    print(df_proposed)
    
    # 14. 데이터프레임을 CSV 파일로 저장
    output_path = 'model_metrics.csv'
    df_proposed.to_csv(output_path)
    print(f"결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    # Generation_list 정의
    Generation_list = ['smote','adasyn','copulagan','ctgan','nbsynthetic']
    main()



    