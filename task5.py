from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lower, col

spark = SparkSession.builder.appName("FakeNewsTask5").getOrCreate()

# Reload and preprocess same as task4
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
df = df.withColumn("text", lower(df["text"]))
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df = tokenizer.transform(df)
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = remover.transform(df)

hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
df = hashingTF.transform(df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df)
df = idf_model.transform(df)

indexer = StringIndexer(inputCol="label", outputCol="label_index")
df = indexer.fit(df).transform(df)

# Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_data)

# Predict
predictions = model.transform(test_data)

# Evaluate
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

# Print + save
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

from pyspark.sql import Row
metrics_df = spark.createDataFrame([
    Row(Metric="Accuracy", Value=round(accuracy, 2)),
    Row(Metric="F1 Score", Value=round(f1, 2))
])
metrics_df.write.csv("task5_output.csv", header=True)
