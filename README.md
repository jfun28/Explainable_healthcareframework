# 🏥 XAI-Powered Childhood Weight Management System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
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

```mermaid
graph TD
   A[IoT Devices] --> B[Data Collection]
   B --> C[Preprocessing Pipeline]
   C --> D[Synthetic Data Generation]
   D --> E[Hybrid XAI Model]
   E --> F[Weight Change Prediction]
   F --> G[SHAP Analysis]
   F --> H[TabNet Masks]
   G --> I[Personalized Insights]
   H --> I
'''

📋 Requirements
python>=3.8
tensorflow>=2.0
xgboost>=1.5.0
pytorch-tabnet>=3.1.1
shap>=0.41.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
nbsynthetic>=0.1.0

📊 Dataset Information
Data Collection

School A (Seoul): 362 elementary students, 6 months, 44,226 records
School B (Jeju): 82 elementary students, 8 weeks, 3,343 records
Devices: Samsung Galaxy Fit 2, WUD! app, Samsung Health

Features Collected
FeatureSourceDescriptionHeight/WeightManual entryRegular updates by participants/parentsCalorie IntakeFood photos/manualBased on National Food Nutrition databaseStep CountSmartwatchDaily physical activitySleep DurationSmartwatch/smartphoneSleep pattern analysisBurned CaloriesSmartwatchEnergy expenditure
Target Labels

Label 1: Weight Loss (1.15%)
Label 2: Weight Maintenance (96.15%)
Label 3: Weight Gain (2.7%)
