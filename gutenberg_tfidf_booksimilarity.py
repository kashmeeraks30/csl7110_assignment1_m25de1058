
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    input_file_name, regexp_extract, col, count, lit,
    lower, regexp_replace, split, explode, size,
    sum as spark_sum, log, row_number, when,
    collect_list, struct
)
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("Gutenberg_TFIDF").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Loading data
books_dataframe = (
    spark.read.text("hdfs:///user/kashmeera/gutenberg", wholetext=True)
    .withColumnRenamed("value", "text")
    .withColumn("file_path", input_file_name())
    .withColumn("file_name", regexp_extract("file_path", r"([^/]+)$", 1))
    .select("file_name", "text")
)
print("Loaded", books_dataframe.count(), "books")

#Preprocessing 
print("\nPREPROCESSING\n")

cleaned = (
    books_dataframe
    .withColumn(
        "clean_text",
        regexp_replace(
            col("text"),
            r"(?s)^.*?\*\*\* START OF THE PROJECT GUTENBERG[^\n]*\n",
            ""
        )
    )
    .withColumn(
        "clean_text",
        regexp_replace(
            col("clean_text"),
            r"(?s)\*\*\* END OF THE PROJECT GUTENBERG.*$",
            ""
        )
    )
    .withColumn("clean_text", lower(col("clean_text")))
    .withColumn("clean_text", regexp_replace(col("clean_text"), r"[^a-z\s]", " "))
    .withColumn("clean_text", regexp_replace(col("clean_text"), r"\s+", " "))
    .select("file_name", "clean_text")
)

# Removing stop words 
STOP_RE = (
    r"\b(a|an|the|and|or|but|in|on|at|to|for|of|with|by|from|as|is|was|are|"
    r"were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|"
    r"might|shall|can|that|this|these|those|it|its|i|you|he|she|we|they|my|"
    r"your|his|her|our|their|me|him|us|them|what|which|who|not|no|so|if|up|"
    r"out|about|into|than|then|when|where|how|all|there|here|more|one|two|"
    r"said|also)\b"
)

words_clean = (
    cleaned
    .withColumn("clean_text", regexp_replace(col("clean_text"), STOP_RE, " "))
    .withColumn("clean_text", regexp_replace(col("clean_text"), r"\s+", " "))
    .withColumn("words", split(col("clean_text"), " "))
    .select("file_name", explode(col("words")).alias("word"))
    # Drop empty strings and tokens <= 2 characters
    .filter((col("word") != "") & (size(split(col("word"), "")) > 2))
)

print("Sample of word counts per book:")
words_clean.groupBy("file_name").agg(count("*").alias("word_count")).show(5)


print("\nTF-IDF CALCULATION\n")

word_counts = (
    words_clean
    .groupBy("file_name", "word")
    .agg(count("*").alias("word_freq"))
)


total_per_book = (
    words_clean
    .groupBy("file_name")
    .agg(count("*").alias("total_words"))
)

tf_df = (
    word_counts
    .join(total_per_book, on="file_name")
    .withColumn("tf", col("word_freq") / col("total_words"))
    .select("file_name", "word", "tf")
)

print("Term Frequency of words:")
tf_df.orderBy(col("tf").desc()).show(10, truncate=False)


N = books_dataframe.count()

doc_freq = (
    word_counts
    .groupBy("word")
    .agg(count("*").alias("doc_freq"))
)

idf_df = (
    doc_freq
    .withColumn("idf", log(lit(float(N + 1)) / (col("doc_freq") + 1)) + 1)
    .select("word", "idf")
)

print("IDF sample:")
idf_df.orderBy(col("idf").desc()).show(10, truncate=False)


tfidf_df = (
    tf_df
    .join(idf_df, on="word")
    .withColumn("tfidf", col("tf") * col("idf"))
    .select("file_name", "word", "tfidf")
)

print("TF-IDF sample:")
tfidf_df.orderBy(col("tfidf").desc()).show(20, truncate=False)


print("\nBOOK SIMILARITY (COSINE)\n")

# Restricting TF-IDF terms to 200 per book
TOP_K = 200

window_spec = Window.partitionBy("file_name").orderBy(col("tfidf").desc())

tfidf_topk = (
    tfidf_df
    .withColumn("rn", row_number().over(window_spec))
    .filter(col("rn") <= TOP_K)
    .drop("rn")
    .cache()
)

print("Each book represented as a vector of its TF-IDF scores")
print("    Each book is a sparse vector: [(word, tfidf_score), ...]\n")

book_vectors = (
    tfidf_topk
    .groupBy("file_name")
    .agg(
        collect_list(struct(col("tfidf"), col("word"))).alias("tfidf_vector")
    )
    .orderBy("file_name")
)

print("Number of books represented as vectors:", book_vectors.count())
print("\nSample — vector for 10.txt:")
book_vectors.filter(col("file_name") == "10.txt").show(1, truncate=False)


pairs = (
    tfidf_topk.alias("a")
    .join(tfidf_topk.alias("b"), on="word")
    .filter(col("a.file_name") < col("b.file_name"))
    .withColumn("dot", col("a.tfidf") * col("b.tfidf"))
    .groupBy(
        col("a.file_name").alias("book_a"),
        col("b.file_name").alias("book_b")
    )
    .agg(spark_sum("dot").alias("dot_product"))
)


norms = (
    tfidf_topk
    .groupBy("file_name")
    .agg(spark_sum(col("tfidf") * col("tfidf")).alias("norm_sq"))
    .withColumn("norm", col("norm_sq") ** 0.5)
    .select("file_name", "norm")
)


similarity = (
    pairs
    .join(norms.withColumnRenamed("file_name", "book_a")
               .withColumnRenamed("norm", "norm_a"), on="book_a")
    .join(norms.withColumnRenamed("file_name", "book_b")
               .withColumnRenamed("norm", "norm_b"), on="book_b")
    .withColumn("cosine_similarity",
                col("dot_product") / (col("norm_a") * col("norm_b")))
    .select("book_a", "book_b", "cosine_similarity")
)

print("Sample cosine similarity scores:")
similarity.orderBy(col("cosine_similarity").desc()).show(20, truncate=False)


query_book = "10.txt"
print(f"\nTop 5 books most similar to '{query_book}':")

top5 = (
    similarity
    .filter((col("book_a") == query_book) | (col("book_b") == query_book))
    .withColumn(
        "other_book",
        when(col("book_a") == query_book, col("book_b")).otherwise(col("book_a"))
    )
    .select("other_book", "cosine_similarity")
    .orderBy(col("cosine_similarity").desc())
    .limit(5)
)
top5.show(truncate=False)


tfidf_topk.unpersist()
spark.stop()
