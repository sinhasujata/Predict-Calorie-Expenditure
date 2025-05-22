# Predict Calorie Expenditure

 A complete machine learning pipeline to estimate calories burned based on biometric and activity features.  
 Built for [Kaggle Playground Series - Season 5, Episode 5](https://www.kaggle.com/competitions/playground-series-s5e5/overview).



## Project Objective

This project was created for the **[Kaggle Playground Series - S5E5](https://www.kaggle.com/competitions/playground-series-s5e5/overview)** competition.  
The goal: **Predict the number of calories burned by an individual during physical activity** based on features like height, weight, heart rate, and duration.

We explored both classical and advanced models, focusing on:
- Deep feature engineering  
- Tree-based ensemble learning  
- Neural networks  
- Hyperparameter tuning  
- Model comparison and interpretation



## Strategy & Workflow

* **Exploratory Data Analysis (EDA)**
* **Baseline Models** – Linear Regression, Random Forest
* **Neural Networks** – Built from scratch and with FastAI  
* **Feature Engineering** – Domain-informed transformations    
* **Advanced Tree Models** – XGBoost, LGBM, CatBoost  
* **Hyperparameter Tuning** – GridSearchCV, RandomizedSearchCV  
* **Evaluation & Comparison** – Cross-validated RMSE



## Notebooks Overview

| Step | Description | Notebook |
|------|-------------|----------|
| 1. | Exploratory analysis of the dataset | [EDA.ipynb](https://github.com/sinhasujata/Predict-Calorie-Expenditure/blob/main/EDA.ipynb) |
| 2. | Linear regression baseline model | [Linear regression.ipynb](https://github.com/sinhasujata/Predict-Calorie-Expenditure/blob/main/Linear%20regression.ipynb) |
| 3. | Neural network built from scratch | [NN from Scratch.ipynb](https://github.com/sinhasujata/Predict-Calorie-Expenditure/blob/main/A%20neural%20network.ipynb) |
| 4. | Neural network using FastAI | [FastAI NN.ipynb](https://github.com/sinhasujata/Predict-Calorie-Expenditure/blob/main/neural%20network%20using%20fastai.ipynb) |
| 5. | Random Forests: Feature importance & interpretation | [Random Forests.ipynb](https://github.com/sinhasujata/Predict-Calorie-Expenditure/blob/main/Random%20Forests.ipynb) |
| 6. | Gradient Boosting models: XGBoost, LGBM, CatBoost | [Ensembles.ipynb](https://github.com/sinhasujata/Predict-Calorie-Expenditure/blob/main/Random%20Forests%2C%20Gradient%20Boosting%2C%20LGBM%20and%20CatBoost.ipynb) |



## Feature Engineering Highlights

- **Domain-driven transformations:**
  - `BMI`, `HeartRate_per_Age`, `Temp_Heart_Interaction`
- **Derived feature**: `Intensity = Heart Rate × Duration`
- **Group statistics**: Mean & std of heart rate by sex
- **Polynomial and log/sqrt transforms** to address skew
- **Interaction terms** and **second-order features**



## Model Performance

| Model                 | Feature Config            | RMSE Score |
|----------------------|---------------------------|------------|
| Linear Regression     | without activation        | 0.38839    |
| Linear Regression     | with Relu activation      | 0.12304    |
| Neural Network        | using log1p for target    | 0.12643    |
| Neural Network         | using fastai and adding parameters | 0.06930    |
| Neural Network         | Using fastai and ensemble  | 0.08295    |
| Random Forest         | With `intensity` feature  | 0.07372    |
| Random Forest         | Without `intensity`       | 0.06463    |
| XGBoost               | Default                   | 0.05818    |
| LightGBM              | Default                   | 0.05816    |
| XGBoost (Tuned)       | Tuned via RandomSearch    | 0.05795    |
| CatBoost (Tuned)      | Tuned via GridSearch      | 0.05795    |



## Deep Learning Attempts

- Built a **neural network from scratch** using PyTorch-like logic
- Trained a **FastAI model** with categorical embeddings and regularization
- Tree-based models outperformed deep learning on this tabular dataset



## Educational References

This project was guided by FastAI’s outstanding educational materials by [Jeremy Howard](https://www.fast.ai/):

- [How Random Forests Really Work](https://www.kaggle.com/code/jhoward/how-random-forests-really-work)  
- [Linear Model and Neural Net from Scratch](https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch)  
- [How Does a Neural Net Really Work?](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work)  
- [Getting Started with NLP (applies many NN basics)](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners)



## Tech Stack

- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn  
- **Models**: XGBoost, LightGBM, CatBoost, RandomForest, FastAI  
- **Tools**: Jupyter Notebook, Kaggle Kernels



## Outcome

- The best models (XGBoost & CatBoost with tuning) achieved an RMSE of **~0.0579**  
- Feature engineering had **significant impact** on performance  
- Tree-based ensemble models were **more effective** than deep neural networks for this tabular problem  
- Learned and applied multiple ML techniques with real-world utility



## Future Experiments which might be impact this project:

- Ensembling of top models  
- SHAP value-based model interpretation  
- Streamlit/FastAPI deployment  
- Integrating AutoML frameworks



## Acknowledgements

Thanks to:
- The **FastAI** and **Kaggle** communities  
- Dataset and competition by **Kaggle Playground Series S5E5**  
- Open-source contributors and researchers in machine learning



