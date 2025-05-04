from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.sql.functions import lower, concat_ws, col

# Start Spark session
spark = SparkSession.builder.appName("FakeNewsTask3").getOrCreate()

# Load and preprocess CSV
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
df = df.withColumn("text", lower(df["text"]))

# Tokenize text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df = tokenizer.transform(df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = remover.transform(df)

# Apply HashingTF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
df = hashingTF.transform(df)

# Apply IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df)
df = idf_model.transform(df)

# Convert labels to numeric
indexer = StringIndexer(inputCol="label", outputCol="label_index")
df = indexer.fit(df).transform(df)

# Fix incompatible output types
df = df.withColumn("filtered_words_str", concat_ws(" ", col("filtered_words")))
df = df.withColumn("features_str", col("features").cast("string"))

# Save final output to CSV
df.select("id", "filtered_words_str", "features_str", "label_index") \
  .write.csv("task3_output.csv", header=True)
