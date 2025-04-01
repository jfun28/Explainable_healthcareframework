# main.py
import pandas as pd
import torch

# 로컬 모듈 임포트
from data_preparation import preprocess_data, convert_to_tensor
from encoding_tabnet import train_tabnet_classifier, extract_tabnet_embeddings
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
        
        # 2. TabNet 모델 학습
        tabnet_clf = train_tabnet_classifier(X_train, y_train, X_valid, y_valid)
        
    
        # 3. 모델 저장
        model_save_path = path+f'\\3.Training_XAI\\model_hist\\{generation}_proposed_tabnetEmbedd.pickle'
        save_model(tabnet_clf, model_save_path)
        
        # 4. 예측
        # prediction = tabnet_clf.predict(X_test)
        
        # 5. 텐서 변환
        X_train_torch, X_test_torch = convert_to_tensor(X_train, X_test)
        
        # 6. TabNet 임베딩 추출
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X_train_feature_attrs = extract_tabnet_embeddings(tabnet_clf, X_train_torch, device=device)
        X_test_feature_attrs = extract_tabnet_embeddings(tabnet_clf, X_test_torch, device=device)
        
        # 7. 표준화3
        X_train_std, X_test_std, scaler = standardize_features(X_train_feature_attrs, X_test_feature_attrs)
        
        # 8. 스케일러 저장
        scaler_save_path = path+f'\\3.Training_XAI\\scaler_hist\\{generation}_proposed_standard_scaler.pickle'
        save_model(scaler, scaler_save_path)
        
        # 9. XGBoost 모델 최적화 및 학습
        results, xgb_pred, best_xgb = optimize_and_compare_models(
            X_train_std, 
            X_test_std, 
            y_train.astype(int), 
            y_test.astype(int), 
            n_trials=10
        )
        # 10. 결과 출력
        print_results(results)
        
        # 11. XGBoost 모델 저장
        xgb_save_path = path+f'\\3.Training_XAI\\model_hist\\{generation}_proposed.pickle'
        save_model(best_xgb, xgb_save_path)
        
        # 12. 성능 지표 계산
        accuracy, f1, precision, recall = calculate_metrics(y_test, xgb_pred)
        
        # 13. 결과를 딕셔너리로 저장
        result_dict = {
            'Model': f"{generation}-proposed",
            'Accuracy': round(accuracy, 5),
            'F1score': round(f1, 5),
            'Precision': round(precision, 5),
            'Recall': round(recall, 5)
        }
        
        # 14. 리스트에 딕셔너리 추가
        model_results_list.append(result_dict)
    
    # 15. 데이터프레임 생성
    df_proposed = pd.DataFrame(model_results_list)
    print("\n결과 요약:")
    print(df_proposed)
    
    # 16. 데이터프레임을 CSV 파일로 저장
    output_path = 'model_metrics.csv'
    df_proposed.to_csv(output_path)
    print(f"결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    # Generation_list 정의
    Generation_list = ['smote','adasyn','copulagan','ctgan','nbsynthetic']
    main()