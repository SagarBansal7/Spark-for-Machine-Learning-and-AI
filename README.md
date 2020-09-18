# Spark for Machine Learning and AI
-	Distributed
-	Processing
-	Big Data
- Modular Architecture

**Use Cases**
-	Real-time monitoring
-	Text analysis
-	Ecommerce pattern analysis
-	Healthcare and genomic analysis

**Steps in ML**
1)	Pre-processing
a.	Extract, transform, and load data to staging area
b.	Review data for missing data and invalid values
c.	Normalize and scale numeric data
d.	Standardize categorical data
2)	Model Building
a.	Selecting algos
b.	Executing algo to fit data to models
c.	Tuning hyperparameters
3)	Validation
a.	Applying model to additional tests

Installation
````
-	Java -version
-	cd ..
-	cd ..
-	dir
-	cd spark
-	cd bin
-	.\pyspark
````
Open file
````
emp_df = spark.read.csv(“”)
emp_df.schema
emp_df.columns
emp_df.take(5) – first five
emp_df.count()
sample_df = emp_df.sample(False, 0.1)
sample_df.count()
emp_mgrs_df = emp_df.filter(“salary >=100000”)
emp_mgrs_df.count()
emp_mgrs_df.select(“salary”).show()
````

**Components of Spark Mlib**
Algos, Workflows, Utilities

**Data Prep and Transformation**
````
Intro to pre-processing

-	Normalize numeric data
o	From original range to 0 to 1
o	from pyspark.ml feature import minmaxscaler
o	from pyspark.ml.linalg import Vectors
o	features_df = spark.createDataFrame([
       (1, Vectors.dense([10.0, 10000.0,  1.0]), ),
       (2, Vectors.dense([20.0, 30000.0,  2.0]), ),
       (3, Vectors.dense([30.0, 40000.0,  3.0]),)
       ], [“id”, “features”])
o	features_df.take(1)
o	feature_scaler = MinMaxScaler(inputCol = “features”, outputCol=”sfeatures”)
o	smodel = feature_scaler.fit(features_df)
o	sfeatures_df = smodel.transform(features_df)
o	sfeatures_df.take(1)
o	sfeatures_df.select(“features”, “sfeatures”).show()
     
-	Standardize numeric data
o	Mao data values from their original range to -1 to 1
o	Mean value of 0
o	Normally distributed with std deviation of 1
o	Used when attributes have different scales and ML algos assumer a normal distribution
o	from pyspark.ml.feature import StandardScaler
o	from pyspark.ml.linalg import Vectors
o	features_df = spark.createDataFrame([
	(1, Vectors.dense([10.0, 10000.0,  1.0]), ),
	(2, Vectors.dense([20.0, 30000.0,  2.0]), ),
	(3, Vectors.dense([30.0, 40000.0,  3.0]),)
	], [“id”, “features”])
o	features_df.take(1)
o	feature_stand_scaler = StandardScaler(inputCol=”features”, outputCol=”sfeatures”, withStd = True, withMean = True)
o	standa_smodel = feature_stand_scaler.fit(feature_df)
o	stand_sfeatures_df = stand_smodel.transform(features_df)
o	stand_sfeatures_df.take(1)
o	stand_sfeatures_df.show()

-	Partitioning numeric data
o	From continuous to buckets
o	Deciles and percentiles are examples of buckets
o	Useful when you want to work with groups of values instead of a continuous range of values
o	from pyspark.ml.feature import Bucketizer
o	splits = [-float(“inf”), -10, 0, 10, float(“inf”)]
o	b_data = [(-800.0), (-10.5,), (-1.7,), (0,), (8.2,), (90.1,)]
o	b_df = spark.createDataFrame(b_data, [“features”])
o	b_df.show()
o	bucketizer = Bucketizer(splits=splits, inputCol=”features”, outputCol=”bfeatures”)
o	bucketed_df = bucketizer.transform(b_df)
o	bucketed_df.show()

-	Text: Tokenizing
o	Single string to a set of token
o	from pyspark.ml.feature import Tokenizer
o	sentences_df = spark.createDataFrame([
	(1,”This is an introduction to Spark MLlib”), 
	(2, “MLlib includes libraries for classifications and regression”),
	(3, “It also contains supporting tools for pipelines”)],
	[“id”, “sentence”])
o	sentences_df.show()
o	sent_token = Tokenizer(inputCol =”sentence”, outputCol=”words”) //col names
o	sent_tokenized_df = sent_token.transform(sentences_df)
o	sent_tokenized_df.show()

-	TF-IDF
o	From a single, long string, to a vector indicating the frequency of each word in a text relative to a group of texts
o	Infrequently used words are more useful for distinguishing categories of text
o	from pyspark.ml.feature import HashingTF

````



