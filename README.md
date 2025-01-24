# MLend Mini-Project: Deception Detection using Audio Features
---
### NOTICE
<i>Note that modification on Figure 4 Title has not been updated properly(Energy variability(x) -> Pitch Consistency(o)).<br>
The refined title should be <b>'Figure 4. Feature importance comparison: before and after adding <u>"Pitch Consistency"</u>'</b></i>

---

## MLEnd Datasets

This project is part of the **ECS7020P: Principles of Machine Learning** module at Queen Mary University of London. The dataset used, the **MLEnd Small Deception Dataset**, is part of the [**MLEnd Datasets**](https://mlenddatasets.github.io/) collection, an educational resource. This project contributes to the practical application of concepts taught in the ECS7020P module by using real-world datasets curated as part of a collaborative educational initiative.
The **MLEnd Datasets** were developed as part of an innovative learning approach in the **Principles of Machine Learning** and **Introduction to Data Science Programming** modules. 
This project specifically focuses on detecting deception in speech by classifying audio recordings as *truthful* or *deceptive*, using features extracted from the **MLEnd Small Deception Dataset**.

---
## Dataset

You can download the dataset from the following sources:
- **Primary Access**: Use the `mlend` Python library (instructions below).
```python
# 1 – Install library - make sure you have version 1.0.0.4
pip install mlend==1.0.0.4

# 2. Import library and functions
import mlend
from mlend import download_deception_small, deception_small_load

# 3. Download small data
datadir = download_deception_small(save_to='MLEnd', subset={}, verbose=1, overwrite=False)

# 4. Read file paths
TrainSet, TestSet, MAPs = deception_small_load(datadir_main=datadir, train_test_split=None, verbose=1, encode_labels=True)

# To read the documentation on the given functions run:
help(download_deception_small)
help(deception_small_load)
```
 
- **Alternative Download Link**: [Google Drive](https://drive.google.com/file/d/1Yf-A07B8R84QfBmKrBi__8HWiVcpzZGU/view)
### Instructions
1. Download the dataset from the link above.
2. Extract the audio files into a folder on your local machine.
3. Specify the path to the audio files in the notebook or script when running the project.
   For example, if you extract the files to a folder named `audio` in the root directory:
   
---

## Introduction
This project aims to classify audio recordings as either *truthful* or *deceptive* stories using machine learning models. The analysis focuses on extracting meaningful audio features, exploring feature variability through clustering, and evaluating supervised learning models for deception detection.

The dataset used is the **MLEnd Deception Dataset**, which contains 100 audio samples of participants recounting truthful and deceptive stories.

---

## Problem Statement
Accurately detecting deception in speech has important applications, including law enforcement, courtroom testimonies, and interrogations. This project investigates:
1. **Feature Variability**: Are there distinguishable patterns in audio features between truthful and deceptive speech?
2. **Model Effectiveness**: How well can machine learning models classify truthful vs. deceptive stories based on these features?

---

## Features Extracted
From each audio sample, the following features were derived:
- **Power**: Signal energy.
- **Pitch Mean** and **Pitch Standard Deviation**: Capturing vocal tone and its variability.
- **Fraction of Voiced Regions**: Indicator of speech fluency.
- **Average Silence Duration**: Cognitive load-related pause patterns.
- **Spectral Flux**: Changes in the audio spectrum over time.
- **Pitch Consistency** (New Feature): A measure of vocal pitch stability, added to improve classification performance.

These features were selected based on prior research in speech analysis and deception detection.

---

## Methodology
1. **Feature Extraction**:
   - Extracted features from 30-second segments of each recording using `librosa`.
   - Stored extracted features and labels as `.npy` files for efficient processing.

2. **Exploratory Analysis**:
   - Used **Pairwise Maximum Overlap-Based Clustering (PMOC)** to explore feature variability.
   - Applied the **Elbow Method** and **Silhouette Scores** to identify meaningful clusters.
   - Introduced the `Pitch Consistency` feature based on cluster analysis insights.

3. **Classification Models**:
   - **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)** were trained and validated.
   - An ensemble model combining KNN and SVM through hard voting was also implemented to enhance performance.

4. **Hyperparameter Tuning**:
   - Optimized SVM hyperparameters (`C`, `gamma`) using grid search with cross-validation.

5. **Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
   - Compared results between initial features and updated features (with `Pitch Consistency`).

---

## Results
- **Best Model**: The ensemble of KNN and SVM achieved a validation accuracy of **67.5%** on non-normalised data.
- Adding the `Pitch Consistency` feature did not significantly enhance model performance, suggesting the need for more advanced feature engineering.
- Test set results revealed limitations, with the ensemble model achieving a test accuracy of **45.5%**, reflecting challenges in generalisation.

---

## Limitations and Future Directions
- **Feature Limitations**: Overlap in certain features (e.g., silence duration) reduced their effectiveness in distinguishing classes.
- **Model Challenges**: Overfitting was observed, with significant drops in performance on unseen data.
- **Future Work**:
  - Investigate high-level features like intonation or semantics for deception detection.
  - Explore advanced ensemble methods (e.g., stacking, boosting).
  - Address class imbalance using techniques like weighted models or resampling.

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/TJK-GB/MLend_MiniProject.git
   cd MLend_MiniProject
   
2. Install dependencies:
   'pip install -r requirements.txt'
   
---

### Note on Features and Labels
- The extracted features (`features_extracted.npy`) and labels (`labels_extracted.npy`) are generated during the feature extraction step in the Jupyter Notebook.
- If these files are not present, users must run the notebook first to generate them.

---

## Technologies Used
- Python (Libraries: scikit-learn, librosa, NumPy, Pandas, Matplotlib)
- Jupyter Notebook for implementation.
- GitHub for version control and sharing.

---

## How to Use
- Run the Jupyter notebook to explore the analysis.
- Use the provided scripts to re-train models or extract features from new datasets.

---

### Special Thanks
I would like to express my gratitude to Professor Jesús Requena Carrión and Dr.Ziquan Liu for their guidance and support throughout the ECS7020P: Principles of Machine Learning module.
Their expertise and feedbacks were invaluable in completing this project.

I also want to extend my appreciation to all the demonstrators who provided assistance during the module.
Their patience and dedication made a significant difference in helping me navigate challenges and successfully complete this work.
