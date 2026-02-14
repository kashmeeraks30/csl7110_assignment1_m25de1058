from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    input_file_name, regexp_extract, length, year, 
    col, count, avg, trim, when, coalesce, lit
)

spark = SparkSession.builder.appName("Gutenberg").getOrCreate()


books_dataframe = spark.read.text("hdfs:///user/kashmeera/gutenberg", wholetext=True) \
    .withColumnRenamed("value", "text") \
    .withColumn("file_path", input_file_name()) \
    .withColumn("file_name", regexp_extract("file_path", r"([^/]+)$", 1)) \
    .select("file_name", "text")

book_count = books_dataframe.count()
print("Loaded " + str(book_count) + " books")

books_dataframe.show(5, truncate=100)


print("\nMETADATA EXTRACTION \n")

#Extracting Title
books_with_metadata = books_dataframe.withColumn(
    "title",
    trim(regexp_extract("text", r"(?i)Title:\s*(.+?)(?:\r?\n|$)", 1))
)

# Extracting Release Date
books_with_metadata = books_with_metadata.withColumn(
    "release_date",
    trim(regexp_extract("text", r"(?i)Release Date:\s*(.+?)(?:\r?\n|\[|$)", 1))
)

# Extracting Language
books_with_metadata = books_with_metadata.withColumn(
    "language",
    trim(regexp_extract("text", r"(?i)Language:\s*(\w+)", 1))
)

# Extracting Encoding
books_with_metadata = books_with_metadata.withColumn(
    "encoding",
    trim(regexp_extract("text", r"(?i)(?:Character set encoding|Charset|Encoding)\s*:\s*([\w-]+)", 1))
)

# Handling missing/empty values
books_with_metadata = books_with_metadata \
    .withColumn("title", when(col("title") == "", None).otherwise(col("title"))) \
    .withColumn("release_date", when(col("release_date") == "", None).otherwise(col("release_date"))) \
    .withColumn("language", when(col("language") == "", None).otherwise(col("language"))) \
    .withColumn("encoding", when(col("encoding") == "", None).otherwise(col("encoding")))

# Showing extracted metadata
print("Samples of Extracted Metadata:")
books_with_metadata.select("file_name", "title", "release_date", "language", "encoding").show(10, truncate=80)

# Check for missing values
print("\n=== MISSING VALUES ANALYSIS ===")
total_books = books_with_metadata.count()
missing_title = books_with_metadata.filter(col("title").isNull()).count()
missing_release_date = books_with_metadata.filter(col("release_date").isNull()).count()
missing_language = books_with_metadata.filter(col("language").isNull()).count()
missing_encoding = books_with_metadata.filter(col("encoding").isNull()).count()

print("Total books: " + str(total_books))
print("Missing title: " + str(missing_title) + " (" + str(round(missing_title*100.0/total_books, 2)) + "%)")
print("Missing release_date: " + str(missing_release_date) + " (" + str(round(missing_release_date*100.0/total_books, 2)) + "%)")
print("Missing language: " + str(missing_language) + " (" + str(round(missing_language*100.0/total_books, 2)) + "%)")
print("Missing encoding: " + str(missing_encoding) + " (" + str(round(missing_encoding*100.0/total_books, 2)) + "%)")

print("\n\nANALYSIS\n")

# 1. Calculate the number of books released each year
print("1. Number of books released each year:")

# Extract year from various date formats
# Handles formats like: "January 1, 2008", "2008", "January 2008", etc.
books_with_year = books_with_metadata.withColumn(
    "release_year",
    regexp_extract("release_date", r"(\d{4})", 1)
).withColumn(
    "release_year",
    when(col("release_year") == "", None).otherwise(col("release_year").cast("int"))
)

books_per_year = books_with_year \
    .filter(col("release_year").isNotNull()) \
    .groupBy("release_year") \
    .agg(count("*").alias("book_count")) \
    .orderBy("release_year")

books_per_year.show(50, truncate=False)

print("\nTop 10 years with most releases:")
books_per_year.orderBy(col("book_count").desc()).show(10, truncate=False)

# 2. Find the most common language in the dataset
print("\n2. Most common language in the dataset:")

language_distribution = books_with_metadata \
    .filter(col("language").isNotNull()) \
    .groupBy("language") \
    .agg(count("*").alias("book_count")) \
    .orderBy(col("book_count").desc())

language_distribution.show(20, truncate=False)

most_common_language = language_distribution.first()
if most_common_language:
    print("\nMost common language: " + str(most_common_language["language"]) +
          " with " + str(most_common_language["book_count"]) + " books")

# 3. Determine the average length of book titles
print("\n3. Average length of book titles (in characters):")

books_with_title_length = books_with_metadata \
    .filter(col("title").isNotNull()) \
    .withColumn("title_length", length("title"))

avg_title_length = books_with_title_length.agg(avg("title_length")).first()[0]
print("Average title length: " + str(round(avg_title_length, 2)) + " characters")

spark.stop()
