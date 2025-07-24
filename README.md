# Application of Explainable Artificial Intelligence for personalized childhood weight management using IoT data

This repository summarizes the research paper "Application of Explainable Artificial Intelligence for personalized childhood weight management using IoT data" by Jaemin Jeong et al. The study proposes a comprehensive framework leveraging wearable devices and AI to address challenges in childhood obesity research, including data limitations, class imbalance, and model interpretability.


**Keywords**: Healthcare, Shapley additive explanation (SHAP), Tabnet, Explainable artificial intelligence (XAI), Generative adversarial networks (GAN)

## 1. Introduction
- Annual increase in the prevalence of childhood obesity
- Current digital healthcare applications have limited effectiveness in motivating users from an obesity prevention perspective

<img src="https://github.com/user-attachments/assets/d1247798-1687-47a4-8966-2e0c9d40d59b" style="width:50%;">

**Research Objective**:
1. Develop a personalized AI healthcare framework
2. Proactively predicts weight fluctuations
3. Performs screening and causal analysis

## 2. Background
**Synthetic Data Generation** 
- Process of creating artificial data that mimics the statistical properties and patterns of real-world data
- Oversampling: Synthetic Miniority Oversampling Technique (SMOTE), Adaptive Synthetic Sampling (ADASYN)
- GAN: CTGAN, CopulaGAN, TableGAN, WGAN

**Explainable Artificial Intelligence (XAI)**
- XAI enhances AI interpretability and trust by using interpretable models or post-hoc explanation methods
- Model-intrinsic methods : TabNet, Decision tree, Linear and logistic regression, Naïve bayes, and General Additive Models (GAMs)
- Model-post hoc methods : SHapley Additive exPlanations (SHAP), Local Interpretable Model-agnostic Explanations (LIME)

**SHapley Additive exPlanations (SHAP)**
- It calculates the importance score of a feature by evaluating how the model prediction changes with and without that feature
<img src="https://github.com/user-attachments/assets/209187c2-404d-4ea2-b230-aac100fa62ba" style="width:50%;">

## 3. Framework for childhood weight management using XAI

### 3.1. Overview
<table>
  <tr>
    <td style="width:30%;">
      <img src="https://github.com/user-attachments/assets/5f2cd6c8-5905-4512-b096-fc3fbe6a6b37" width="70%" />
    </td>
    <td style="vertical-align:top; padding-left: 10px;">
      <ol>
        <li><strong>Data Collection and Preprocessing</strong><br/>
            - Collect data through IoT devices<br/>
            - Implement a preprocessing pipeline
        </li><br/>
        <li><strong>Training of Explainable AI Model</strong><br/>
            - Generate training data using diverse synthetic methods<br/>
            - Train a proposed XAI model
        </li><br/>
        <li><strong>Model Evaluation and Explanation</strong><br/>
            - Evaluate model performance using external dataset<br/>
            - Apply XAI techniques as proof of concept
        </li>
      </ol>
    </td>
  </tr>
</table>


### 3.2. Data collection and Preprocessing
[cite_start]Data collected from wearable devices and smartphones are synchronized and managed by health-tracking applications[cite: 148]. [cite_start]Height and weight data are regularly updated by participants or their parents[cite: 153]. [cite_start]Caloric intake is estimated based on food photos or manual entries, referencing the National Food Nutrition database[cite: 154, 155, 157]. [cite_start]Burned calories, step count, and sleep time are collected from the smartwatch or inferred from smartphone sensors[cite: 165, 221, 222, 223, 224].

**Preprocessing steps**:
1.  **Remove outlier values**: Outliers in height were corrected with historical values. Calorie intake outliers were adjusted by multiplying single-meal logs by three and applying min-max scaling with an upper limit of 4,000 kcal. [cite_start]Other features like step count, sleep duration, and burned calories also had upper limits applied. [cite: 228, 229, 232, 233, 234, 235, 236, 237, 238, 239, 242]
2.  **Missing value imputation**: Linear interpolation was used for height and weight due to their linear changes over time. [cite_start]Other missing values were replaced with the median for each user. [cite: 243, 244, 245, 246, 247, 249]
3.  **Data grouping & Labeling**: Input data was averaged over a 14-day period daily. [cite_start]Weight changes exceeding a 100-gram threshold were categorized into three classes: 1 (weight loss), 2 (weight maintenance), and 3 (weight gain). [cite: 251, 252, 253, 254, 256, 257, 259, 260, 365, 366]

### 3.3. Training XAI models with synthetic data generation

### 3.3.1. Synthetic data generation
[cite_start]To address class imbalance, 10,000 data points were generated for each minority class (weight loss and weight gain) and incorporated into the training dataset[cite: 263, 264, 272]. [cite_start]Various data generation techniques, including SMOTE, ADASYN, nbsynthetic, CTGAN, and CopulaGAN, were compared[cite: 265, 267]. [cite_start]Nbsynthetic was chosen for its superior effectiveness in model performance and robustness, being an unconditional Wasserstein GAN-based open-source library[cite: 274, 275].

### 3.3.2. Proposed hybrid XAI model architecture
[cite_start]A multiclass prediction model for weight change (loss, maintenance, gain) was developed using a hybrid approach combining TabNet and XGBoost[cite: 279, 280]. [cite_start]TabNet performs sparse feature selection and trains attention masks, revealing features focused on at each step, thereby enhancing interpretability[cite: 281, 282, 283, 306]. [cite_start]XGBoost, an enhanced gradient boosting algorithm, reduces training error incrementally[cite: 307, 308]. [cite_start]This hybrid architecture effectively combines TabNet's interpretability with XGBoost's predictive power[cite: 332]. [cite_start]The model used five TabNet steps repeated for 100 epochs, and XGBoost parameters were optimized using Bayesian optimization[cite: 333].

## 3.4. Model evaluation and explanation
[cite_start]The model was evaluated using 10% of School A's data as test data and School B's data as external validation data[cite: 335, 336]. [cite_start]Performance metrics included accuracy, F1-score, precision, and recall[cite: 337]. [cite_start]TabNet's intrinsic mask mechanism and SHAP analysis were used for interpretability[cite: 338, 339]. [cite_start]Case studies demonstrated the practical validity of the model[cite: 341].

## 4. Experimental results

### 4.1. Description of experimental dataset
[cite_start]Lifelog data was collected from 362 elementary school students in Seoul (School A) for six months and 82 students in Jeju (School B) for eight weeks, using Samsung Galaxy Fit 2 smartwatches and synchronized applications (WUD! and Samsung Health)[cite: 350, 351, 353, 355, 357, 358]. [cite_start]The final dataset for School A had 44,226 records from 243 participants, and for School B, 3,343 records from 77 participants[cite: 356, 359]. [cite_start]Weight changes exceeding 100 grams were considered clinically meaningful[cite: 365]. [cite_start]Weight loss (Label 1) was 1.15%, weight gain (Label 3) was 2.7%, and weight maintenance (Label 2) was 96.15%, indicating class imbalance[cite: 367, 368].

### 4.2. Model evaluation

#### 4.2.1. Results of model prediction by generation methods
[cite_start]Training the model on data augmented with synthetic samples significantly reduced prediction bias towards the weight maintenance class[cite: 377, 378]. [cite_start]The model trained with nbsynthetic-generated data achieved the most balanced performance on the test dataset, with class-wise accuracies of 0.9673 (weight loss), 0.9955 (weight maintenance), and 0.9268 (weight gain)[cite: 379]. [cite_start]Nbsynthetic-based generation also demonstrated the best overall performance on the test dataset (accuracy 0.980, F1-score 0.979, precision 0.979, recall 0.980) and strong resilience on the external dataset (accuracy 0.852, F1-score 0.814)[cite: 381, 483].

#### 4.2.2. Results of predictions by the interpretable model
[cite_start]The proposed hybrid model consistently outperformed other interpretable models (TabNet, decision tree, naive Bayes, LDA, GAMs) across all evaluation metrics on both test and external datasets when trained with nbsynthetic-generated data[cite: 598, 599, 600, 603, 604].

#### 4.2.3. Statistical validation of performance differences among models
[cite_start]Statistical analysis showed that the performance differences between the proposed XAI model and comparison models were statistically significant ($p < 0.05$) for most metrics[cite: 606, 610]. [cite_start]However, for Decision Tree and TabNet models, some metrics (F1-score, precision, recall) had p-values exceeding 0.05, indicating less statistical significance in those specific differences[cite: 611, 612].

### 4.3. Explainability results

### 4.3.1. Model explanations
[cite_start]**Global feature importance using TabNet (model-intrinsic)**: The most influential features were height, step count, weight, calorie intake, sleep duration, and burned calories, in descending order[cite: 691].
[cite_start]**Global feature importance using SHAP (post-hoc)**: SHAP values showed similar rankings, with height having the highest contribution and burned calories the lowest[cite: 697, 698]. [cite_start]Height was prominent due to the study participants being elementary school children in their growth phase, suggesting growth-related factors contribute to weight gain[cite: 699, 766, 767]. [cite_start]The framework helps distinguish between unhealthy weight gain and normal growth[cite: 770].

### 4.3.2. Proof of concept for personalized healthcare
Case studies for weight loss and weight gain demonstrated the framework's practical application and personalized insights.

**Weight loss (User 1, days 20-24)**:
* [cite_start]**TabNet mask heatmap**: Weight, step count, and burned calories had the most significant influence on weight loss prediction[cite: 788, 790].
* [cite_start]**SHAP force and waterfall plots**: Calorie intake, weight, sleep time, height, and step count contributed positively to weight loss prediction, with calorie intake and sleep duration being the most important[cite: 795, 797, 800, 801]. [cite_start]A marked decrease in calorie intake and an increase in sleep duration explained the predicted weight loss[cite: 802, 803].

**Weight gain (User 2, days 41-45)**:
* [cite_start]**TabNet mask heatmap**: Burned calorie count, step count, and height were identified as primary contributors to weight gain prediction[cite: 923, 924].
* [cite_start]**SHAP force and waterfall plots**: Sleep duration and step count had the highest SHAP values, indicating their significant influence on weight gain prediction[cite: 928, 929]. [cite_start]Reduced sleep duration and a decrease in step count contributed to weight gain[cite: 970, 971, 972].

## 5. Discussion
[cite_start]This study successfully proposed an explainable healthcare framework for early childhood obesity prevention by predicting weight changes using IoT data[cite: 974, 975]. [cite_start]The nbsynthetic-based Wasserstein GAN effectively augmented minority class samples, leading to superior performance of the hybrid TabNet-XGBoost model, even on external datasets[cite: 976, 977, 978, 979]. [cite_start]XAI techniques (TabNet masks and SHAP) provided meaningful feedback by interpreting decision-making processes and analyzing lifestyle patterns[cite: 980, 981].

**Limitations and Future Work**:
* Frequent missing data due to the requirement of carrying devices, limiting the full utilization of certain variables. [cite_start]Future work should enhance incentive systems for user participation[cite: 983, 984, 985, 986].
* [cite_start]Performance decrease on external datasets highlights the need for further optimization techniques like domain adaptation or transfer learning to improve generalization[cite: 987, 988, 989].
* [cite_start]Discrepancies between variable importance from mask attention and SHAP values suggest a need for an integrated framework for model interpretation[cite: 990, 991].

## 6. Conclusion
[cite_start]The proposed framework accurately predicts weight changes and effectively interprets model predictions using XAI techniques (TabNet mask mechanism and SHAP)[cite: 994]. [cite_start]The hybrid TabNet-XGBoost architecture, combined with nbsynthetic-generated synthetic data, consistently showed high performance[cite: 995, 996, 1079]. [cite_start]The consistency between the two interpretability methods reinforces the framework's reliability[cite: 1082]. [cite_start]Case studies validated real-world applicability, showing individualized predictions and interpretations aligning with varying lifestyle patterns[cite: 1083, 1084]. [cite_start]This explainable predictive healthcare framework has the potential to enhance user trust, expand healthcare applications, and contribute significantly to childhood obesity prevention through personalized feedback and early awareness[cite: 1086, 1087, 1088].
