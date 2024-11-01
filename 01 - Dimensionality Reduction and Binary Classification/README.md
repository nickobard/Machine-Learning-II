# Dimensionality Reduction and Binary Classification

The notebook addresses the task of classifying images into two categories: **Trousers (label 0)** and **Dress (label 1)**. The images are 28x28 pixels in grayscale, sourced from the Fashion MNIST dataset. The primary objectives are to preprocess the data, explore various classification models, apply dimensionality reduction techniques, and select the best-performing model to predict labels on new data.

---

### **Data Exploration**

- **Data Loading**: The dataset is loaded from `train.csv`.
- **Data Splitting**: The data is split into training and validation sets using an 60/40 split.
- **Visualization**:
    - Displayed random samples of images along with their pixel value histograms.
    - Observed that images can be differentiated by pixel positions, while their histograms of pixel values are similar.
    - Analyzed the mean images for both classes individually and combined.

**Observations**:
- The mean image across both classes resembles the Trousers class more.
- The distribution of classes is balanced with a slight inclination towards Trousers.

---

### **Classification Without Dimensionality Reduction**

#### **1. Support Vector Machine (SVM)**

- **Model Suitability**: SVM is effective for high-dimensional data and can handle non-linear separability using kernel functions.
- **Hyperparameters Tuned**:
    - `C`: Regularization parameter.
    - `kernel`: Types tried include linear, polynomial (`poly`), and radial basis function (`rbf`).
    - `gamma`: Kernel coefficient for `rbf` and `poly` kernels.
    - `degree`: Degree of the polynomial kernel function.
- **Training Approach**:
    - Used `GridSearchCV` with 5-fold cross-validation for hyperparameter tuning.
    - Trained models on original data, normalized data, and standardized data.
- **Results**:
    - **Best Model Parameters**:
        - `kernel`: `'poly'`
        - `degree`: `2`
        - `C`: `20`
        - `gamma`: `'scale'`
    - **Cross-Validation Score**: Approximately **98%** accuracy.
    - **Effect of Scaling**: Minimal impact on performance; results were similar across original, normalized, and standardized data.

#### **2. Naive Bayes Classifier**

- **Model Suitability**: Naive Bayes is robust against the curse of dimensionality and is suitable for high-dimensional data.
- **Types Explored**:
    - **Bernoulli Naive Bayes**: Assumes binary features.
        - **Hyperparameters**:
            - `alpha`: Smoothing parameter.
            - `binarize`: Threshold for binarizing feature values.
    - **Multinomial Naive Bayes**: Suitable for discrete count features.
        - **Hyperparameter**:
            - `alpha`: Smoothing parameter.
- **Training Approach**:
    - Trained on original and normalized data.
    - Used `GridSearchCV` for hyperparameter tuning.
- **Results**:
    - **BernoulliNB**:
        - Performance was sensitive to the `binarize` parameter.
        - Scaling required adjusting the `binarize` threshold accordingly.
    - **MultinomialNB**:
        - Did not outperform SVM.
    - **Overall**: Naive Bayes models achieved lower accuracy compared to SVM.

#### **3. Linear Discriminant Analysis (LDA)**

- **Model Suitability**: LDA is a linear classifier that can be effective for classification tasks with normally distributed classes.
- **Hyperparameters Tuned**:
    - `solver`: Methods tried include `'svd'` and `'lsqr'`.
- **Training Approach**:
    - Trained on original, normalized, and standardized data.
    - Used `GridSearchCV` for hyperparameter tuning.
- **Results**:
    - **Cross-Validation Score**: Lower than SVM.
    - **Effect of Scaling**: Minimal impact on performance.
    - **Overall**: LDA did not perform as well as SVM.

---

### **Dimensionality Reduction with PCA**

- **Motivation**: Reduce dimensionality to eliminate features (pixels) with near-zero variance, potentially improving model performance and computational efficiency.
- **Process**:
    - Applied Principal Component Analysis (PCA) to the training data.
    - Analyzed the explained variance of each principal component.
- **Findings**:
    - **190 Components** have near-zero variance and could be removed without significant loss of information.
- **Outcome**:
    - Did not proceed with applying models on PCA-reduced data, focusing instead on the original feature set.

---

### **Final Model Selection**

- **Chosen Model**: **Support Vector Machine (SVM)** with a polynomial kernel.
- **Final Model Parameters**:
    - `kernel`: `'poly'`
    - `degree`: `2`
    - `C`: `20`
    - `gamma`: `'scale'`
- **Training**:
    - Retrained the SVM model on the combined training and validation data for maximum utilization of available data.
- **Estimated Performance**:
    - Expected accuracy on new, unseen data is **at least 95%**, based on cross-validation results.

---

### **Prediction on Evaluation Data**

- **Data Loading**: Loaded test images from `evaluate.csv` (without labels).
- **Preprocessing**: Applied the same preprocessing steps as the training data (no scaling required).
- **Prediction**:
    - Used the final SVM model to predict labels for the evaluation dataset.
- **Results Saving**:
    - Created `results.csv` containing two columns:
        - `ID`: Identifier for each image.
        - `label`: Predicted class label (`0` for Trousers, `1` for Dress).
- **File Structure**:
    - Organized saved models and results within a structured directory for reproducibility and clarity.

---

### **Summary of Findings**

- **Model Performance**:
    - **SVM** outperformed both Naive Bayes and LDA in this binary classification task.
    - **Scaling** (normalization or standardization) did not significantly impact model performance due to the consistent scaling of pixel values in the dataset.
- **Dimensionality Reduction**:
    - **PCA** revealed that many features had near-zero variance.
    - Despite this, the best results were obtained without reducing the dimensionality, likely because the SVM model effectively handled the high-dimensional space.
- **Model Selection Rationale**:
    - The SVM model with a polynomial kernel provided the best balance between accuracy and computational efficiency.
    - Cross-validation ensured the model's generalizability to unseen data.

---

### **Conclusion**

The notebook successfully demonstrates the process of:

- **Data Exploration**: Understanding the dataset through visualization and statistical analysis.
- **Model Training and Evaluation**: Applying different classification algorithms and tuning their hyperparameters.
- **Dimensionality Reduction**: Exploring PCA to identify and potentially remove redundant features.
- **Model Selection**: Choosing the best-performing model based on cross-validation scores.
- **Prediction on New Data**: Using the final model to make predictions on unseen data and preparing the results for submission.

**Final Deliverables**:

- A trained **SVM classifier** capable of distinguishing between Trousers and Dress images with high accuracy.
- A `results.csv` file containing the predicted labels for the evaluation dataset.

**Note**: The directory structure includes saved models and results, ensuring reproducibility and facilitating further analysis if needed.

---

**File Structure**:

```
.
├── evaluate.csv
├── homework_01_B232.ipynb
├── README.md
├── results.csv
├── train.csv
└── train_results
    ├── lda
    │   ├── normalized_rd-42.pickle
    │   ├── original_rd-42.pickle
    │   └── standardized_rd-42.pickle
    ├── naive_bayes
    │   ├── multinomial_original_rd-42.pickle
    │   ├── normalized_rd-42.pickle
    │   ├── original_rd-42.pickle
    │   └── standardized_rd-42.pickle
    └── svc
        ├── linear_normalization_rd-42.pickle
        ├── linear_original_rd-42.pickle
        ├── normalization_rd-42.pickle
        ├── original_rd-42.pickle
        └── standardization_rd-42.pickle

5 directories, 17 files
```
