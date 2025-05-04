from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import lower, concat_ws

# Start Spark session
spark = SparkSession.builder.appName("FakeNewsTask2").getOrCreate()

# Load CSV
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Lowercase the text column
df = df.withColumn("text", lower(df["text"]))

# Tokenize text into words
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized_df = tokenizer.transform(df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = remover.transform(tokenized_df)

# Convert filtered_words array to a string for CSV export
cleaned_df = cleaned_df.withColumn("filtered_words_str", concat_ws(" ", cleaned_df["filtered_words"]))

# Select final columns and write to CSV
cleaned_df.select("id", "title", "filtered_words_str", "label") \
          .write.csv("task2_output.csv", header=True)
