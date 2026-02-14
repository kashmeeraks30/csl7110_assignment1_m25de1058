
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    input_file_name, regexp_extract, col, count, lit,
    trim, when, lower, abs as spark_abs,
    year as spark_year, to_date, coalesce
)

spark = SparkSession.builder.appName("Gutenberg_AuthorNetwork").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


print("\nLOADING BOOKS\n")

books_raw = (
    spark.read.text("hdfs:///user/kashmeera/gutenberg", wholetext=True)
    .withColumnRenamed("value", "text")
    .withColumn("file_path", input_file_name())
    .withColumn("file_name", regexp_extract("file_path", r"([^/]+)$", 1))
    .select("file_name", "text")
)

print("Number of Books loaded:", books_raw.count())


print("\nPreprocessing\n")


books_meta = books_raw.withColumn(
    "author_raw",
    trim(regexp_extract("text", r"(?i)Authors?\s*:\s*(.+?)(?:\r?\n|$)", 1))
)

books_meta = books_meta.withColumn(
    "author",
    when(
        col("author_raw").rlike(
            r"(?i)^(anonymous|unknown|various|n/a|none|\s*)$"
        ),
        lit(None)
    ).otherwise(col("author_raw"))
)


books_meta = books_meta.withColumn(
    "release_date_raw",
    trim(regexp_extract("text", r"(?i)Release Date\s*:\s*(.+?)(?:\r?\n|\[|$)", 1))
).withColumn(
    "release_year",
    regexp_extract("release_date_raw", r"(\d{4})", 1)
).withColumn(
    "release_year",
    when(col("release_year") == "", lit(None))
    .otherwise(col("release_year").cast("int"))
)


books_valid = (
    books_meta
    .filter(col("author").isNotNull() & col("release_year").isNotNull())
    .select("file_name", "author", "release_year")
    .distinct()            
)

total_valid = books_valid.count()
print(f"Books with both author and release year: {total_valid}")

print("\nSample of preprocessed data:")
books_valid.show(15, truncate=60)


total_books   = books_raw.count()
missing_auth  = books_meta.filter(col("author").isNull()).count()
missing_year  = books_meta.filter(col("release_year").isNull()).count()
missing_both  = books_meta.filter(
    col("author").isNull() | col("release_year").isNull()
).count()

#print("\nMissing values")
#print(f"Total books          : {total_books}")
#print(f"Missing author       : {missing_auth}  ({round(missing_auth*100/total_books,1)}%)")
#print(f"Missing release year : {missing_year}  ({round(missing_year*100/total_books,1)}%)")
#print(f"Excluded (either)    : {missing_both}  ({round(missing_both*100/total_books,1)}%)")



print("\nINFLUENCE NETWORK CONSTRUCTION \n")

X_YEARS = 5        

print(f"Influence Range: X = {X_YEARS} years")
#print("Edge definition : author_A → author_B  if  0 < year_B − year_A <= X\n")


from pyspark.sql.functions import min as spark_min

author_years = (
    books_valid
    .groupBy("author")
    .agg(spark_min("release_year").alias("first_year"))
)

#print(f"Distinct authors with a release year: {author_years.count()}")
#print("Sample author years:")
#author_years.orderBy("first_year").show(10, truncate=50)

edges = (
    author_years.alias("a")
    .join(author_years.alias("b"),
          (col("b.first_year") - col("a.first_year") > 0) &
          (col("b.first_year") - col("a.first_year") <= lit(X_YEARS)))
    .select(
        col("a.author").alias("influencer"),      
        col("b.author").alias("influenced"),       
        col("a.first_year").alias("influencer_year"),
        col("b.first_year").alias("influenced_year"),
        (col("b.first_year") - col("a.first_year")).alias("year_gap")
    )
)

edge_count = edges.count()
print(f"\nTotal directed edges in influence network: {edge_count:,}")
print("\nSample edges (influencer → influenced):")
edges.orderBy("influencer_year").show(15, truncate=50)


print("\nAnalysis\n")

out_degree = (
    edges
    .groupBy("influencer")
    .agg(count("influenced").alias("out_degree"))
    .withColumnRenamed("influencer", "author")
)


in_degree = (
    edges
    .groupBy("influenced")
    .agg(count("influencer").alias("in_degree"))
    .withColumnRenamed("influenced", "author")
)


all_authors = author_years.select("author")

degrees = (
    all_authors
    .join(out_degree, on="author", how="left")
    .join(in_degree,  on="author", how="left")
    .fillna(0, subset=["out_degree", "in_degree"])
    .withColumn("total_degree", col("out_degree") + col("in_degree"))
    .orderBy(col("total_degree").desc())
)

print("Degree table:")
degrees.show(20, truncate=50)


from pyspark.sql.functions import avg, max as spark_max, min as spark_min2

print("\nDegree Summary")
degrees.select(
    avg("in_degree").alias("avg_in_degree"),
    avg("out_degree").alias("avg_out_degree"),
    spark_max("in_degree").alias("max_in_degree"),
    spark_max("out_degree").alias("max_out_degree"),
).show(truncate=False)


print("Top 5 authors with highest in-degree")

top5_in = (
    degrees
    .select("author", "in_degree", "out_degree")
    .orderBy(col("in_degree").desc())
    .limit(5)
)
top5_in.show(truncate=60)


print("\nTop 5 authors with highest out degree\n")

top5_out = (
    degrees
    .select("author", "out_degree", "in_degree")
    .orderBy(col("out_degree").desc())
    .limit(5)
)
top5_out.show(truncate=60)


print("Adjusting X to explore different influence ranges:\n")
print(f"{'X (years)':<12} {'Edges':>12} {'Avg in-degree':>15} {'Avg out-degree':>16}")

for x in [1, 2, 5, 10, 20]:
    e = (
        author_years.alias("a")
        .join(author_years.alias("b"),
              (col("b.first_year") - col("a.first_year") > 0) &
              (col("b.first_year") - col("a.first_year") <= lit(x)))
        .count()
    )
    n = author_years.count()
    avg_deg = round(e / n, 2) if n > 0 else 0
    marker = " <-- used above" if x == X_YEARS else ""
    print(f"{x:<12} {e:>12,} {avg_deg:>15} {avg_deg:>16}{marker}")

print()

spark.stop()

