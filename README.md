# Spark-iForest

Isolation Forest (iForest) is an effective model that focuses on anomaly isolation. 
iForest uses tree structure for modeling data, iTree isolates anomalies closer to the root of the tree as compared to normal points. 
A anomaly score is calculated by iForest model to measure the abnormality of the data instances. The lower, the more abnormal.

More details about iForest can be found in the following papers: 
<a href="https://dl.acm.org/citation.cfm?id=1511387">Isolation Forest</a> [1] 
and <a href="https://dl.acm.org/citation.cfm?id=2133363">Isolation-Based Anomaly Detection</a> [2].

We design and implement a distributed iForest on Spark, which is trained via model-wise parallelism, and predicts a new Dataset via data-wise parallelism. 
It is implemented in the following steps:
  1. Sampling data from a Dataset. Data instances are sampled and grouped for each iTree. 
  As indicated in the paper, the number samples for constructing each tree is usually not very large (default value 256). 
  Thus we can construct sampled paired RDD, where each row key is tree index and row value is a group of sampled data instances for a tree.
  1. Training and constructing each iTree on parallel via a map operation and collect all iTrees to construct a iForest model.
  1. Predict a new Dataset on parallel via a map operation with the collected iForest model.

## Usage

Spark iForest is designed and implemented easy to use. The usage is similar to the iForest sklearn implementation [3]. 

*Parameters:*

- *numTrees:* The number of trees in the iforest model (>0).
- *maxSamples:* The number of samples to draw from data to train each tree (>0).
If maxSamples <= 1, the algorithm will draw maxSamples * totalSample samples.
If maxSamples > 1, the algorithm will draw maxSamples samples.
The total memory is about maxSamples * numTrees * 4 + maxSamples * 8 bytes.
- *maxFeatures:* The number of features to draw from data to train each tree (>0).
If maxFeatures <= 1, the algorithm will draw maxFeatures * totalFeatures features.
If maxFeatures > 1, the algorithm will draw maxFeatures features.
- *maxDepth:* The height limit used in constructing a tree (>0).
The default value will be about log2(numSamples).
- *contamination:* The proportion of outliers in the data set, the value should be in (0, 1).
It is only used in the prediction phase to convert anomaly score to predicted labels. 
In order to enhance performance, Our method to get anomaly score threshold is caculated by approxQuantile.
Note that this is an approximate quantiles computation, if you want an exactly answer,
you can extract ”$anomalyScoreCol" to select your anomalies.
- *bootstrap:* If true, individual trees are fit on random subsets of the training data sampled with replacement.
If false, sampling without replacement is performed.
- *seed:* The seed used by the randam number generator.
- *featuresCol:* features column name, default "features".
- *anomalyScoreCol:* Anomaly score column name, default "anomalyScore".
- *predictionCol:* Prediction column name, default "prediction".



## Examples

The following codes are an example for detecting anamaly data points using 
Wisconsin Breast Cancer (Breastw) Dataset [4].

*Scala API* 
```scala
val spark = SparkSession
.builder()
.master("local") // test in local mode
.appName("iforest example")
.getOrCreate()

val startTime = System.currentTimeMillis()

// Wisconsin Breast Cancer Dataset
val dataset = spark.read.option("inferSchema", "true")
.csv("data/anomaly-detection/breastw.csv")

// Index label values: 2 -> 0, 4 -> 1
val indexer = new StringIndexer()
.setInputCol("_c10")
.setOutputCol("label")

val assembler = new VectorAssembler()
assembler.setInputCols(Array("_c1", "_c2", "_c3", "_c4", "_c5", "_c6", "_c7", "_c8", "_c9"))
assembler.setOutputCol("features")

val iForest = new IForest()
.setNumTrees(100)
.setMaxSamples(256)
.setContamination(0.35)
.setBootstrap(false)
.setMaxDepth(100)
.setSeed(123456L)

val pipeline = new Pipeline().setStages(Array(indexer, assembler, iForest))
val model = pipeline.fit(dataset)
val predictions = model.transform(dataset)

val binaryMetrics = new BinaryClassificationMetrics(
predictions.select("prediction", "label").rdd.map {
case Row(label: Double, ground: Double) => (label, ground)
}
)

val endTime = System.currentTimeMillis()
println(s"Training and predicting time: ${(endTime - startTime) / 1000} seconds.")
println(s"The model's auc: ${binaryMetrics.areaUnderROC()}")
```

## Benchmark
TO BE ADDED


## Requirements

Spark-iForest is built on Spark 2.1.1 or later version.

## Build From Source

`mvn clean package`

## Licenses

Spark-IForest is available under Apache Licenses 2.0.

## Acknowledgement

Spark iForest is designed and implemented together with my former intern Fang, Jie at Transwarp (transwarp.io). 
Thanks for his great contribution. In addition, thanks for the supports of Discover Team.

## Contact and Feedback

If you encounter any bugs, feel free to submit an issue or pull request. Also you can email to:
<a href="fangzhou.yang@hotmail.com">Yang, Fangzhou (fangzhou.yang@hotmail.com)</a>

## References:

[1] Liu F T, Ting K M, Zhou Z, et al. Isolation Forest[C]. international conference on data mining, 2008.

[2] Liu F T, Ting K M, Zhou Z, et al. Isolation-Based Anomaly Detection[J]. ACM Transactions on Knowledge Discovery From Data, 2012, 6(1).

[3] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

[4] A. Asuncion and D. Newman. UCI machine learning repository, 2007.
