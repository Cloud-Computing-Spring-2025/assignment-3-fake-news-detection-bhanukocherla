# Assignment-5-FakeNews-Detection
This project builds a machine learning pipeline to classify news articles as **FAKE** or **REAL** using Apache Spark.
## 📂 Dataset

- `fake_news_sample.csv` — A sample dataset containing fake and real news titles and texts with labels.
To generate the dataset:

python Dataset_Generator.py
Task 1: Load & Basic Exploration
- Load CSV and infer schema
- Create a temporary view
- Display first 5 rows
- Count total articles
- Show distinct labels
- **Output**: `task1_output.csv`

```bash
python3 task1.py


### Task 2: Text Preprocessing
Convert text to lowercase

Tokenize text into words

Remove stopwords

Convert token array to string for CSV compatibility

- **Output**: `task2_output.csv`


```bash

python3 task2.py

Task 3: Feature Extraction
TF-IDF vectorization using HashingTF and IDF

Encode label column using StringIndexer

Convert vectors and arrays to strings for saving

Output: task3_output.csv
```bash

python3 task3.py

Task 4: Model Training
Split data into 80% train / 20% test

Train LogisticRegression model

Predict on test set

Save predictions

Output: task4_output.csv
```bash

python3 task4.py

Task 5: Evaluation
Evaluate with MulticlassClassificationEvaluator

Compute Accuracy and F1 Score

Save metrics to CSV

Output: task5_output.csv
```bash

python3 task5.py

## How to Run
Install Dependencies
pip install pyspark faker pandas

Generate Dataset
python Dataset_Generator.py

Run Main Spark Pipeline
python Main.py




