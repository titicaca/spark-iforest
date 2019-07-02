# Spark-iForest
[![Build Status](https://travis-ci.org/titicaca/spark-iforest.svg?branch=master)](https://travis-ci.org/titicaca/spark-iforest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Isolation Forest (iForest) is an effective model that focuses on anomaly isolation. 
iForest uses tree structure for modeling data, iTree isolates anomalies closer to the root of the tree as compared to normal points. 
A anomaly score is calculated by iForest model to measure the abnormality of the data instances. The higher, the more abnormal.

More details about iForest can be found in the following papers: 
<a href="https://dl.acm.org/citation.cfm?id=1511387">Isolation Forest</a> [1] 
and <a href="https://dl.acm.org/citation.cfm?id=2133363">Isolation-Based Anomaly Detection</a> [2].

We design and implement a distributed iForest on Spark, which is trained via model-wise parallelism, and predicts a new Dataset via data-wise parallelism. 
It is implemented in the following steps:
  1. Sampling data from a Dataset. Data instances are sampled and grouped for each iTree. 
  As indicated in the paper, the number of samples for constructing each tree is usually not very large (default value 256). 
  Thus we can construct sampled paired RDD, where each row key is tree index and row value is a group of sampled data instances for a tree.
  1. Training and constructing each iTree on parallel via a map operation and collect all iTrees to construct a iForest model.
  1. Predict a new Dataset on parallel via a map operation with the collected iForest model.
  

## Install

Step 1. Package spark-iforest jar and deploy it into spark lib

```bash
cd spark-iforest/

mvn clean package -DskipTests

cp target/spark-iforest-<version>.jar $SPARK_HOME/jars/
```

Step 2. Package pyspark-iforest and install it via pip, skip this step if you don't need the python pkg

```bash
cd spark-iforest/python

python setup.py sdist

pip install dist/pyspark-iforest-<version>.tar.gz
```

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
You can set the param approxQuantileRelativeError greater than 0,
in order to calculate an approximate quantile threshold of anomaly scores for large dataset.
- *approxQuantileRelativeError:* Relative Error for Approximate Quantile Calculation (0 <= value <= 1),
default is 0 for calculating the exact value, which would be expensive for large datasets.
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

// Save pipeline model
model.write.overwrite().save("/tmp/iforest.model")

// Load pipeline model
val loadedPipelineModel = PipelineModel.load("/tmp/iforest.model")
// Get loaded iforest model
val loadedIforestModel = loadedPipelineModel.stages(2).asInstanceOf[IForestModel]
println(s"The loaded iforest model has no summary: model.hasSummary = ${loadedIforestModel.hasSummary}")

val binaryMetrics = new BinaryClassificationMetrics(
predictions.select("prediction", "label").rdd.map {
case Row(label: Double, ground: Double) => (label, ground)
}
)

val endTime = System.currentTimeMillis()
println(s"Training and predicting time: ${(endTime - startTime) / 1000} seconds.")
println(s"The model's auc: ${binaryMetrics.areaUnderROC()}")
```

*Python API*
```python
from pyspark.ml.linalg import Vectors
import tempfile


spark = SparkSession \
        .builder.master("local[*]") \
        .appName("IForestExample") \
        .getOrCreate()

data = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([7.0, 9.0]),),
        (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)]

df = spark.createDataFrame(data, ["features"])

from pyspark_iforest.ml.iforest import *

# Init an IForest Object
iforest = IForest(contamination=0.3, maxDepth=2)

# Fit on a given data frame
model = iforest.fit(df)

# Check if the model has summary or not, the newly trained model has the summary info
model.hasSummary

# Show model summary
summary = model.summary

# Show the number of anomalies
summary.numAnomalies

# Predict for a new data frame based on the fitted model
transformed = model.transform(df)

# Collect spark data frame into local df
rows = transformed.collect()

temp_path = tempfile.mkdtemp()
iforest_path = temp_path + "/iforest"

# Save the iforest estimator into the path
iforest.save(iforest_path)

# Load iforest estimator from a path
loaded_iforest = IForest.load(iforest_path)

model_path = temp_path + "/iforest_model"

# Save the fitted model into the model path
model.save(model_path)

# Load a fitted model from a model path
loaded_model = IForestModel.load(model_path)

# The loaded model has no summary info
loaded_model.hasSummary

# Use the loaded model to predict a new data frame
loaded_model.transform(df).show()
```

## Benchmark

#### Environment

Hardware Setup:
- CPU: Intel(R) Xeon(R) ES-2620 V2 @ 2.1GHz
- RAM: 128G

Software Setup:
- Spark Version: v2.2.0
- Sklearn Version: v0.19.1


#### Accuracy Performance

The following table shows the testing AUC result among origin paper [1], spark-iforest and sklearn-iforest.


| Dataset   | #Samples | Anomaly-Rate | Dimension | Origin-Paper | Spark-iForest | Sklearn-iForest |
| ----------|:---------:| -----------:| ---------:| ------------:| -------------:| ---------------:|
| breastw   | 683       | 35%         | 9         | 0.98         | 0.96          | 0.94            |
| shuttle   | 49097     | 7%          | 9         | 1.00         | 0.89          | 0.95            |
| http      | 567498    | 0.4%        | 3         | 1.00         | 0.99          | 0.99            |
| ionosphere| 351       | 36%         | 32        | 0.85         | 0.65          | 0.71            |
| satellite | 6435      | 33%         | 36        | 0.71         | 0.60          | 0.68            |


#### Time Performance

The following table shows the time consuming between sklearn-iforest and spark-iforest.
Here we use the above largest dataset *http* for testing.

|time cost (s) | sklearn     | spark (4 cores) |
|-------------:| -----------:| ---------------:|
| training     | 335         | 34              |
| prediction   | 300         | 86              |

* Model Parameters: numTrees = 100, maxSamples = 256


#### Scalability Performance

The following table shows the scalability of spark-iforest model. The testing dataset is still *http*. 
The memory is set 1G per executor on Spark. The number of cores are range from 1 to 4 cores.    

|time cost (s) | 1 core      | 2 cores      | 3 cores      | 4 cores      |
|-------------:| -----------:| ------------:| ------------:| ------------:|
| training     | 74          | 52           | 40           | 34           |
| prediction   | 272         | 157          | 117          | 86           |

* Model Parameters: numTrees = 100, maxSamples = 256


## Requirements

Spark-iForest is built on Spark 2.4.0 or later version.

## Licenses

Spark-IForest is available under Apache Licenses 2.0.

## Acknowledgement

Spark iForest is designed and implemented together with my former intern Fang, Jie at Transwarp (transwarp.io). 
Thanks for his great contribution. In addition, thanks for the supports of Discover Team.

## Contact and Feedback

If you encounter any bugs, feel free to submit an issue or pull request. Also you can email to:
<a href="fangzhou.yang@hotmail.com">Yang, Fangzhou (fangzhou.yang@hotmail.com)</a>

## Citation

Please cite spark-iforest in your publications if it helped your research. Here is an example BibTeX entry:

```
@misc{titicacasparkiforest,
  title={spark-iforest},
  author={Fangzhou Yang and contributors},
  year={2018},
  publisher={GitHub},
  howpublished={\url{https://github.com/titicaca/spark-iforest}},
}
```

## References:

[1] Liu F T, Ting K M, Zhou Z, et al. Isolation Forest[C]. international conference on data mining, 2008.

[2] Liu F T, Ting K M, Zhou Z, et al. Isolation-Based Anomaly Detection[J]. ACM Transactions on Knowledge Discovery From Data, 2012, 6(1).

[3] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

[4] A. Asuncion and D. Newman. UCI machine learning repository, 2007.
