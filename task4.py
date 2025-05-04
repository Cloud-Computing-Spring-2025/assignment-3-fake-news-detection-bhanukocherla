from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.appName("FakeNewsTask4").getOrCreate()

# Load and preprocess again (for standalone run)
from pyspark.sql.functions import lower
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

# Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_data)

# Predict
predictions = model.transform(test_data)
predictions.select("id", "title", "label_index", "prediction").write.csv("task4_output.csv", header=True)
