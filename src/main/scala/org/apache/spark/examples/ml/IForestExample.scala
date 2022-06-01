package org.apache.spark.examples.ml

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.iforest.{IForest, IForestModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.{Row, SparkSession}

/**
  * An example demonstrating IForest.
  * Run with
  * {{{
  * ./spark-sumbit ...
  * }}}
  */
object IForestExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
        .builder()
        .master("local") // test in local mode
        .appName("iforest example")
        .getOrCreate()

    val startTime = System.currentTimeMillis()

    // Dataset from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
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
  }
}

// scalastyle:on println