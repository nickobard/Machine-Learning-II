Two tasks were solved in this repository:
 - 01 - Dimensionality Reduction and Binary Classification.
 - 02 - Multi-label classification using Feed-Forward Neural Network.

# 01 - Dimensionality Reduction and Binary Classification

The notebook classifies 28x28 grayscale images into Trousers (label 0) and Dress (label 1). Data is split into training and validation sets after exploration and visualization. Three models were applied without dimensionality reduction:

- **Support Vector Machine (SVM):** Achieved approximately **98%** accuracy using a polynomial kernel (`degree=2`, `C=20`). Scaling data had minimal impact.
- **Naive Bayes:** Performed worse than SVM; sensitive to hyperparameters.
- **Linear Discriminant Analysis (LDA):** Less effective than SVM.

Although PCA identified 190 components with near-zero variance, the best results were without dimensionality reduction. The final model, an SVM with a polynomial kernel, is expected to achieve at least **95%** accuracy on new data. Predictions were saved in `results.csv`.

---

# 02 - Neural Networks

The notebook constructs neural networks for classifying 32x32 grayscale images into 10 fashion categories. After data exploration confirming balanced classes, a Feed-Forward Neural Network (FFNN) was developed:

- **Architecture:** Best performance with 3-4 hidden layers of 128-256 units.
- **Regularization:** Techniques like dropout and weight decay did not significantly improve results.
- **Optimizers:** The Adam optimizer outperformed SGD.
- **Normalization:** Adversely affected model performance.

The final model achieved approximately **85%** validation accuracy, with an expected test accuracy around **83%**. Predictions on new data were saved in `results.csv`.

--- 

File structure (excluded concrete log files):

```
.
├── 01 - Dimensionality Reduction and Binary Classification
│   ├── evaluate.csv
│   ├── homework_01_B232.ipynb
│   ├── README.md
│   ├── results.csv
│   ├── train.csv
│   └── train_results
│       ├── lda
│       │   ├── normalized_rd-42.pickle
│       │   ├── original_rd-42.pickle
│       │   └── standardized_rd-42.pickle
│       ├── naive_bayes
│       │   ├── multinomial_original_rd-42.pickle
│       │   ├── normalized_rd-42.pickle
│       │   ├── original_rd-42.pickle
│       │   └── standardized_rd-42.pickle
│       └── svc
│           ├── linear_normalization_rd-42.pickle
│           ├── linear_original_rd-42.pickle
│           ├── normalization_rd-42.pickle
│           ├── original_rd-42.pickle
│           └── standardization_rd-42.pickle
├── 02 - Neural Networks
│   ├── evaluate.csv
│   ├── homework_02_B232.ipynb
│   ├── images
│   │   ├── depth_size_exp
│   │   │   ├── all_rd_42.png
│   │   │   ├── best_model_acc.png
│   │   │   └── best_model_loss.png
│   │   ├── evaluation
│   │   │   └── best_model.png
│   │   ├── ffnn_normalization
│   │   │   ├── all_minmax.png
│   │   │   ├── all_none.png
│   │   │   ├── all_standardization.png
│   │   │   └── weight_decay_test
│   │   │       ├── all_no_wd.png
│   │   │       └── all_wd_0.1.png
│   │   ├── ffnn_optimizers
│   │   │   ├── all_adam.png
│   │   │   ├── all.png
│   │   │   ├── all_sgd.png
│   │   │   └── SGD_spikes.png
│   │   ├── ffnn_regularization
│   │   │   ├── all.png
│   │   │   ├── best_model_acc.png
│   │   │   └── best_model_loss.png
│   │   └── parameter_space_spikes.png
│   ├── logs
│   │   ├── archive
│   │   ├── depth_and_size_experimentation
│   │   ├── ffnn_normalization
│   │   │   └── weight_decay_test
│   │   ├── ffnn_optimizers
│   │   ├── ffnn_regularization
│   │   ├── final
│   │   ├── results.csv
│   │   └── test
│   ├── README.md
│   ├── results.csv
│   ├── train.csv
│   └── utils
│       └── svg_to_png.py
└── README.md

24 directories, 42 files
```