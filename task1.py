from pyspark.sql import SparkSession

# Start Spark session
spark = SparkSession.builder.appName("FakeNewsTask1").getOrCreate()

# Load CSV
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Create temporary view
df.createOrReplaceTempView("news_data")

# Show first 5 rows
df.show(5)

# Count total number of articles
print("Total articles:", df.count())

# Distinct labels
df.select("label").distinct().show()

# Save first 5 rows to CSV
df.limit(5).write.csv("task1_output.csv", header=True)
