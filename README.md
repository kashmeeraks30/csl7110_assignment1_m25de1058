MapReduce:
-----------
WordCount MapReduce Java code is present in WordCount.java

Instructions to run the WordCount java code:
Step 1: javac -source 1.8 -target 1.8 -classpath $(hadoop classpath) -d wordcount_classes WordCount.java
Step 2: jar -cvf WordCount.jar -C wordcount_classes/ .
Step 3: hadoop jar WordCount.jar WordCount /user/kashmeera/wordcount/input/ /user/kashmeera/wordcount/output (input files are present in hdfs at the specified location)


Spark:
-------
To start all the Hadoop Daemons like NameNode, DataNode, ResourceManager, NodeManager:
/opt/hadoop/sbin/start-all.sh

Spark Python code is present in gutenberg_metadata_extraction_analysis.py, gutenberg_tfidf_booksimilarity.py, gutenberg_author_influencenetwork.py for questions 10, 11, 12 respectively
Instructions to run the Spark Python codes:
spark-submit gutenberg_metadata_extraction_analysis.py
spark-submit gutenberg_tfidf_booksimilarity.py
spark-submit gutenberg_author_influencenetwork.py

