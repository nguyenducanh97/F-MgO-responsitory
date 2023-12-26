# Affiliation

This repository contains code for the paper entitled "Integrated Adsorption using Ultrahigh-porosity Magnesium Oxide and Multi-output Predictive Deep Belief Network: A Transformative Approach for fluoride treatment" authored by Duc Anh Nguyen, Viet Bac Nguyen, and Am Jang from the Department of Global Smart City, Electrical and Computer Engineering, Sungkyunkwan University (SKKU), 2066, Seobu-ro, Jangan-gu, Suwon-si, Gyeonggi-do 16419, Republic of Korea. The published article can be found on [].

## Introduction

The rapid growth of the semiconductor industry has resulted in the urgent issue of fluoride (F) contamination in wastewater. In this research, we demonstrate the exceptional potential of Ranunculus-like MgO calcined at 400-600°C (M4-M6) as superior adsorbents for F removal and predict the adsorption performance using several machine learning models. A data frame with 318 rows and 19 columns was created from 954 experimental laboratory tests. In other words, each data point was averaged from triplicated results to avoid overfitting issues. The first 15 columns of the data frame represent input features, including initial pH, dosage of adsorbent, contact time, initial adsorbate concentration, temperature, and the concentration of individual and mixed co-existing components (NO3–, Cl–, HCO3–, SO42–, PO43–, humic acid, K+, Na+, NH4+, COD). The remaining four columns correspond to output features, namely adsorption capacity, removal efficiency, final pH, and leaching Mg. Eight traditional machine learning models (Multiple Linear Regression, ElasticNet Regression, Random Forest, Extra Trees, Lasso, Ridge, BaggingRegressor, KNeighborsRegressor), two deep neural network models (DNN ver 1 and DNN ver 2), and five Deep Belief Network models (DBN ver 1, DBN ver 2, DBN ver 3, DBN ver 4, DBN ver 5) were examined and selected the ones with higher performances for further use.

## Files in this Repository

- `<dbn>`: Library for Deep Belief Network models.
- `<F-removal-by-MgO-data-321data-points.csv>`: Dataset applied in this study.
- `<DBN-model>` and `<DNN-ML-save-model>`: Models after being trained using the prepared dataset.
- `<FeIm for DBN>` and `<PDP data for DBN ver 5>`: Feature importance and partial dependence plot data for DBN models, respectively.
- `<DBN regression for F-removal-by-MgO-data.py>`: DBN, feature importance, and SHAP beeswarm plot algorithms.
- `<DNN and traditional ML regressions for F-removal-by-MgO-data.ipynb>`: DNN, traditional machine learning, and feature importance algorithms.
- `<PDP for DBN regression using F-removal-by-MgO-data.py>`: The partial dependence plot (PDP) algorithms.
- `<Robustness test of DBNv5>`: Prediction performance under different training and testing datasets.

## Findings

Predictive modeling initially falls short, with Lasso and K-NearestNeighbor showing poor accuracy. However, ensemble algorithms (RandomForest, ExtraTrees, Bagging) present improved performance, albeit with limitations in predicting removal efficiency and Mg leakage. Deep Neural Network emerges as a promising approach, achieving good accuracy in predicting Qt and final pH. Yet, the true revelation lies in the application of Deep Belief Network (DBN), a generative model composed of stacked restricted Boltzmann machines, which delivers unparalleled prediction performance (MAE=0.9186, RMSE=2.1406, R2=0.9981) across all output features compared to other studies using over 1000 samples (4 times higher). This captivating DBN model holds immense potential for effectively predicting diverse F effluent types, paving the way for transformative applications in the near future.

## Correspondence

- Email: nguyenducanh@g.skku.edu
- Phone: +82-10-2816-9711 (Korea)

Thank you very much.
