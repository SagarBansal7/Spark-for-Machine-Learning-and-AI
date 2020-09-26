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
    *	Extract, transform, and load data to staging area
    *	Review data for missing data and invalid values
    *	Normalize and scale numeric data
    *	Standardize categorical data
2)	Model Building
    *	Selecting algos
    *	Executing algo to fit data to models
    *	Tuning hyperparameters
3)	Validation
    *	Applying model to additional tests

**Installation**
````cmd
-	Java -version
-	cd ..
-	cd ..
-	dir
-	cd spark
-	cd bin
-	.\pyspark
````
**Open file**
````python
emp_df = spark.read.csv(“file path”)
emp_df.schema #returns the schema
emp_df.columns # returns the columns
emp_df.take(5) #first five
emp_df.count()
sample_df = emp_df.sample(False, 0.1)
sample_df.count()

#Filter all employees with salary greater than 100000
emp_mgrs_df = emp_df.filter(“salary >=100000”)
#Count all employees with salary greater than 100000
emp_mgrs_df.count()
emp_mgrs_df.select(“salary”).show()
````

**Components of Spark Mlib**
* Algos
* Workflows
* Utilities

## Data Prep and Transformation
### Intro to pre-processing

1) Normalize numeric data
````python
from original range to 0 to 1
from pyspark.ml feature import minmaxscaler
from pyspark.ml.linalg import Vectors
features_df = spark.createDataFrame([
    (1, Vectors.dense([10.0, 10000.0,  1.0]), ),
    (2, Vectors.dense([20.0, 30000.0,  2.0]), ),
    (3, Vectors.dense([30.0, 40000.0,  3.0]),)
    ], [“id”, “features”])
features_df.take(1)
feature_scaler = MinMaxScaler(inputCol = “features”, outputCol=”sfeatures”)
smodel = feature_scaler.fit(features_df)
sfeatures_df = smodel.transform(features_df)
sfeatures_df.take(1)
sfeatures_df.select(“features”, “sfeatures”).show()
````
2)	Standardize numeric data
* Make data values from their original range to -1 to 1.
* Mean value of 0.
* Normally distributed with std deviation of 1.
* Used when attributes have different scales and ML algorithms assumes a normal distribution.

````python
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
features_df = spark.createDataFrame([
	(1, Vectors.dense([10.0, 10000.0,  1.0]), ),
	(2, Vectors.dense([20.0, 30000.0,  2.0]), ),
	(3, Vectors.dense([30.0, 40000.0,  3.0]),)
	], [“id”, “features”])
features_df.take(1)
feature_stand_scaler = StandardScaler(inputCol=”features”, outputCol=”sfeatures”, withStd = True, withMean = True)
standa_smodel = feature_stand_scaler.fit(feature_df)
stand_sfeatures_df = stand_smodel.transform(features_df)
stand_sfeatures_df.take(1)
stand_sfeatures_df.show()
````

3)	Partitioning numeric data
*	From continuous to buckets of data.
*	Deciles and percentiles are examples of buckets.
*	Useful when you want to work with groups of values instead of a continuous range of values.

````python
from pyspark.ml.feature import Bucketizer
splits = [-float(“inf”), -10, 0, 10, float(“inf”)]
b_data = [(-800.0), (-10.5,), (-1.7,), (0,), (8.2,), (90.1,)]
b_df = spark.createDataFrame(b_data, [“features”])
b_df.show()
bucketizer = Bucketizer(splits=splits, inputCol=”features”, outputCol=”bfeatures”)
bucketed_df = bucketizer.transform(b_df)
bucketed_df.show()
````

4)	Text: Tokenizing
*	Single string to a set of token.

````python
from pyspark.ml.feature import Tokenizer
sentences_df = spark.createDataFrame([
	(1,”This is an introduction to Spark MLlib”), 
	(2, “MLlib includes libraries for classifications and regression”),
	(3, “It also contains supporting tools for pipelines”)],
	[“id”, “sentence”])
sentences_df.show()
sent_token = Tokenizer(inputCol =”sentence”, outputCol=”words”) //col names
sent_tokenized_df = sent_token.transform(sentences_df)
sent_tokenized_df.show()
````

5)	TF-IDF
*	From a single, long string, to a vector indicating the frequency of each word in a text relative to a group of texts.
*	Infrequently used words are more useful for distinguishing categories of text.

````python
from pyspark.ml.feature import HashingTF, IDF
hashingTF = HashingTF(inputCol="words", outputCol = "rawFeatures", numFeatures=20)
sent_hfTF_df = hashingTF.transform(sent_tokenized_df)
sent_hfTF_df.take(1)
idf = IDF(inputCol="rawFeatures", outputCol="idf_features")
idfModel = idf.fit(sent_hfTF_df)
ifidf_df = idfModel.transform(sent_hfTF_df)
tfidf_df.take(1)
````

## Clustering
* K-means Clustering
* Hierarchical Clustering

## Classification
* Preprocessing
* Naive Bayes Classification
* Decision tree Classification
* Multilayer perceptron Classification

````python
from pyspark.ml.classification import MultiplayerPerceptronCLassifier
#first layer has the same number of nodes as the number of inputs
#last layer has the same number of nodes as the number of outputs
layers = [4,5,5,3]
mlp = MultilayerPerceptronClassifier(layers=layers, seed=1)
mlp_model = mlp.fit(train_df)
mlp_predicitons = mlp_model.transform(test_df)
mlp_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
mlp_accuracy = mlp_evaluator.evaluate(mlp_predictions)
mlp_accuracy
````

## Regression
* Preprocessing
* Linear Regression
* Decision Tree Regression
* Gradient-boosted Tree Regression

````python
from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol="features", labelCol="PE")
gbt_model = gbt.fit(train_df)
gbt_model = gbt_model.transform(test_df)
gbt_evaluator = RegressionEvaluator(labelCol="PE", predictionCol="prediction", metricName="rmse")
gbt_rmse = gbt_evaluator.evaluate(gbt_predictions)
gbt_rmse
````

## Recommendations
* Collaborative Filtering
* Content-Based Filtering

### Spark MLlib supports collaborative filtering

1) Preprocessing - Collaborative Filtering
* Alternating least squares.
* Import ALS from pyspark.ml.recommendation.
* Build a datafarme of user-item ratings.

2) Create an ALS object
* UserCol
* itemCol
* ratingCol

3) Train model using fit

4) Validation
* Create predictions using a transform of an ALS model using test data.
* Create a RegressionEvaluator object.
* Evaluate predictions using the evaluate function of the RegressionEvaluator.

## Process Summary
1) Preprocessing
* Load data into DF.
* Include headers, or column names, in text files.
* Use inferSchema=True.
* Use VectorAssembler to create feature vectors.
* Use StringIndexer to map from string to numeric indexes.

2) Building Models
* Split data into training and test sets.
* Fit models using training data.
* Create predictions by applying a transform to the test data.

3) Validating Models
* Use MLib evaluators (MulticlassClassificationEvaluator, RegressionEvaluator).
* Experiment with multiple algorithms.
* Vary hyperparamenters.

#### For future, explore:
* MLlib Docs
* Kaggle
* AWS Data Sets






