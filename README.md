# 🏥 XAI-based Childhood Weight Management System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pytorch](https://img.shields.io/badge/Pytorch1.16+-orange.svg)](https://pytorch.org)
[![DOI](https://img.shields.io/badge/DOI-10.1234%2Fexample-blue)](https://doi.org/10.1234/example)

> **Application of Explainable Artificial Intelligence for personalized childhood weight management using IoT data**

An innovative framework leveraging wearable devices and explainable AI to address childhood obesity through personalized predictions and interpretable insights.

## 🌟 Key Features

- **📊 IoT Data Integration**: Seamless collection from Samsung Galaxy Fit 2 and smartphones
- **🤖 Hybrid AI Model**: TabNet + XGBoost architecture for superior performance
- **🔍 Explainable AI**: SHAP and TabNet mask mechanisms for interpretability
- **⚖️ Synthetic Data Generation**: Advanced GAN-based techniques for class balance
- **👶 Child-Focused**: Specifically designed for elementary school children
- **📱 Real-time Monitoring**: Continuous lifestyle pattern analysis

## 🏗️ System Architecture
<img alt="image" src="https://github.com/user-attachments/assets/1f492658-d0bb-471d-828f-f763f3a5fc68" width="50%" /> 
'''

## 📋 Requirements
python>=3.8
tensorflow>=2.0
xgboost>=1.5.0
pytorch-tabnet>=3.1.1
shap>=0.41.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
nbsynthetic>=0.1.0

## 📊 Dataset Information

### Data Collection
- **School A (Seoul)**: 362 elementary students, 6 months, 44,226 records
- **School B (Jeju)**: 82 elementary students, 8 weeks, 3,343 records
- **Devices**: Samsung Galaxy Fit 2, WUD! app, Samsung Health

### Features Collected
| Feature | Source | Description |
|---------|--------|-------------|
| Height/Weight | Manual entry | Regular updates by participants/parents |
| Calorie Intake | Food photos/manual | Based on National Food Nutrition database |
| Step Count | Smartwatch | Daily physical activity |
| Sleep Duration | Smartwatch/smartphone | Sleep pattern analysis |
| Burned Calories | Smartwatch | Energy expenditure |

### Target Labels
- **Label 1**: Weight Loss (1.15%)
- **Label 2**: Weight Maintenance (96.15%) 
- **Label 3**: Weight Gain (2.7%)

## 🧠 Model Architecture
Hybrid TabNet-XGBoost Model

<img alt="image" src="https://github.com/user-attachments/assets/50ac2c50-5e9b-4971-96c4-42bd1c1e1913" width="80%"/>

## 📈 Performance Results

### Model Performance (Test Dataset)
<img alt="image" src="https://github.com/user-attachments/assets/5ab89260-1bb0-4c2d-a7a8-f5e0ca20b5b2" width="80%" />


### Class-wise Performance
- **Weight Loss**: 96.73% accuracy
- **Weight Maintenance**: 99.55% accuracy  
- **Weight Gain**: 92.68% accuracy

### External Validation (School B)
- **Accuracy**: 85.2%
- **F1-Score**: 81.4%
