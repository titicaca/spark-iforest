package org.apache.spark.ml.iforest

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.util.Random


class IForestSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  @transient var dataset: Dataset[_] = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    dataset = IForestSuite.generateIForestData(spark, 10, 2)
  }

  test("default parameters") {
    val iforest = new IForest()

    assert(iforest.getNumTrees === 100)
    assert(iforest.getMaxSamples === 1.0)
    assert(iforest.getMaxFeatures === 1.0)
    assert(iforest.getContamination === 0.1)
    assert(!iforest.getBootstrap)
    assert(iforest.getFeaturesCol === "features")
    assert(iforest.getPredictionCol === "prediction")
    assert(iforest.getLabelCol === "label")
    assert(iforest.getAnomalyScoreCol === "anomalyScore")
  }

  test("set parameters") {
    val iforest = new IForest()
        .setNumTrees(10)
        .setMaxSamples(10)
        .setMaxFeatures(10)
        .setMaxDepth(4)
        .setContamination(0.01)
        .setBootstrap(true)
        .setSeed(123L)
        .setFeaturesCol("test_features")
        .setPredictionCol("test_prediction")
        .setLabelCol("test_label")
        .setAnomalyScoreCol("test_anomalyScore")

    assert(iforest.getNumTrees === 10)
    assert(iforest.getMaxSamples === 10)
    assert(iforest.getMaxFeatures === 10)
    assert(iforest.getMaxDepth === 4)
    assert(iforest.getContamination === 0.01)
    assert(iforest.getBootstrap)
    assert(iforest.getSeed === 123L)
    assert(iforest.getFeaturesCol === "test_features")
    assert(iforest.getPredictionCol === "test_prediction")
    assert(iforest.getLabelCol === "test_label")
    assert(iforest.getAnomalyScoreCol === "test_anomalyScore")
  }

  test("split data") {
    // test with bootsrap
    val iforest1 = new IForest()
        .setNumTrees(2)
        .setMaxDepth(4)
        .setMaxSamples(1.0)
        .setMaxFeatures(1.0)
        .setBootstrap(false)
    val rdd1 = iforest1.splitData(dataset)
    val arr = rdd1.map(elem => elem._2).collect
    assert(arr.length === 2 && arr(0) === arr(1))

    // test without bootstrap
    val iforest2 = new IForest()
        .setNumTrees(2)
        .setMaxDepth(4)
        .setMaxSamples(1.0)
        .setMaxFeatures(1.0)
        .setBootstrap(true)
    val rdd2 = iforest2.splitData(dataset)
    val arr2 = rdd1.map(elem => elem._2).collect
    assert(arr.length === 2 && arr(0) === arr(1))
  }

  test("sample features") {
    val data = IForestSuite.generateIVectorArray(4, 3)
    val iforest = new IForest().setSeed(123456L)
    val (sampleResult, featureIdxArr) = iforest.sampleFeatures(data, 4)
    assert(sampleResult.length === 4 && sampleResult(0).length === 3 &&
        sampleResult(1).length === 3 && sampleResult(2).length === 3)
    assert(featureIdxArr.length === 3 && featureIdxArr(0) === 0 && featureIdxArr(1) === 1 && featureIdxArr(2) === 2)

    val (sampleResult2, featureIdxArr2) = iforest.sampleFeatures(data, 2)
    assert(sampleResult2.length === 4 && sampleResult2(0).length === 2 &&
        sampleResult2(1).length === 2 && sampleResult2(2).length === 2)
    assert(featureIdxArr2.length === 2)
  }

  test("fit, transform and summary") {
    val predictionColName = "test_prediction"
    val anomalyScoreName = "test_anomalyScore"
    val iforest = new IForest()
        .setNumTrees(10)
        .setMaxDepth(4)
        .setPredictionCol(predictionColName)
        .setAnomalyScoreCol(anomalyScoreName)
        .setContamination(0.2)
        .setMaxFeatures(0.5)
        .setSeed(123L)
    val model = iforest.fit(dataset)
    assert(model.trees.length === 10)

    val summary = model.summary
    val anomalies = summary.anomalies.collect
    assert(anomalies.length === 10)
    assert(summary.numAnomalies === 2)

    val transformed = model.transform(dataset)
    val expectedColumns = Array("features", predictionColName, anomalyScoreName)
    expectedColumns.foreach { column =>
      assert(transformed.columns.contains(column))
    }
  }

  test("copy estimator and model") {
    val iforest1 = new IForest().setMaxDepth(4)
    val iforest2 = iforest1.copy(ParamMap.empty)
    iforest1.params.foreach { p =>
      if (iforest1.isDefined(p)) {
        (iforest1.getOrDefault(p), iforest2.getOrDefault(p)) match {
          case (Array(values), Array(newValues)) =>
            assert(values === newValues, s"Values do not match on param ${p.name}.")
          case (value, newValue) =>
            assert(value === newValue, s"Values do not match on param ${p.name}.")
        }
      } else {
        assert(!iforest2.isDefined(p), s"Param ${p.name} shouldn't be defined.")
      }
    }

    val model1 = iforest1.fit(dataset)
    val model2 = model1.copy(ParamMap.empty)
    model1.params.foreach { p =>
      if (model1.isDefined(p)) {
        (model1.getOrDefault(p), model2.getOrDefault(p)) match {
          case (Array(values), Array(newValues)) =>
            assert(values === newValues, s"Values do not match on param ${p.name}.")
          case (value, newValue) =>
            assert(value === newValue, s"Values do not match on param ${p.name}.")
        }
      } else {
        assert(!model2.isDefined(p), s"Param ${p.name} shouldn't be defined.")
      }
    }
    assert(model1.summary.featuresCol === model2.summary.featuresCol)
    assert(model1.summary.predictionCol === model2.summary.predictionCol)
    assert(model1.summary.anomalyScoreCol === model2.summary.anomalyScoreCol)
  }

  // test for model read/write
  test("read/write") {
    def checkTreeNodes(node: IFNode, node2: IFNode): Unit = {
      (node, node2) match {
        case (node: IFInternalNode, node2: IFInternalNode) =>
          assert(node.featureValue === node2.featureValue)
          assert(node.featureIndex === node2.featureIndex)
          checkTreeNodes(node.leftChild, node2.leftChild)
          checkTreeNodes(node.rightChild, node2.rightChild)
        case (node: IFLeafNode, node2: IFLeafNode) =>
          assert(node.numInstance === node2.numInstance)
        case _ =>
          throw new AssertionError("Found mismatched nodes")
      }
    }
    def checkModelData(model: IForestModel, model2: IForestModel): Unit = {
      val trees = model.trees
      val trees2 = model2.trees
      assert(trees.length === trees2.length)
      try {
        trees.zip(trees2).foreach { case (node, node2) =>
          checkTreeNodes(node, node2)
        }
      } catch {
        case ex: Exception => throw new AssertionError(
          "checkModelData failed since the two trees were not identical.\n"
        )
      }
    }

    val iforest = new IForest()
    testEstimatorAndModelReadWrite(
      iforest,
      dataset,
      IForestSuite.allParamSettings,
      IForestSuite.allParamSettings,
      checkModelData
    )
  }

  test("boundary case") {
    intercept[IllegalArgumentException] {
      new IForest().setMaxSamples(-1)
    }

    intercept[IllegalArgumentException] {
      new IForest().setMaxFeatures(-1)
    }

    intercept[IllegalArgumentException] {
      new IForest().setMaxDepth(-1)
    }

    intercept[IllegalArgumentException] {
      new IForest().setContamination(-1)
    }

    intercept[IllegalArgumentException] {
      val iforest = new IForest()
          .setMaxSamples(20)
      iforest.fit(dataset)
    }
  }
}

object IForestSuite {
  case class TestRow(features: Vector)

  def generateIForestData(spark: SparkSession, rows: Int, dim: Int): DataFrame = {
    val sc = spark.sparkContext
    val rdd = sc.parallelize(1 to rows).map(i => Vectors.dense(Array.fill(dim)(i.toDouble)))
        .map(v => TestRow(v))
    spark.createDataFrame(rdd)
  }
  case class dataSchema(features: Vector, label: Double)

  def generateDataWithLabel(spark: SparkSession): DataFrame = {

    val data = Seq(
      dataSchema(Vectors.dense(Array(-1.0, 0.0)), 0.0),
      dataSchema(Vectors.dense(Array(-1.0, 1.0)), 0.0),
      dataSchema(Vectors.dense(Array(0.0, 1.0)), 0.0),
      dataSchema(Vectors.dense(Array(1.0, 1.0)), 0.0),
      dataSchema(Vectors.dense(Array(1.0, 0.0)), 0.0),
      dataSchema(Vectors.dense(Array(1.0, 1.0)), 0.0),
      dataSchema(Vectors.dense(Array(0.0, -1.0)), 0.0),
      dataSchema(Vectors.dense(Array(-1.0, -1.0)), 0.0),
      dataSchema(Vectors.dense(Array(5.0, 5.0)), 1.0),
      dataSchema(Vectors.dense(Array(-5.0, -5.0)), 1.0)
    )
    val sc = spark.sparkContext
    spark.createDataFrame(sc.parallelize(data))
  }

  def generateIVectorArray(row: Int, col: Int, seed: Long = 100L): Array[Vector] = {
    val rand = new Random(seed)
    Array.tabulate(row) { i =>
      Vectors.dense(Array.fill(col)(i.toDouble * rand.nextInt(10)))
    }
  }

  val allParamSettings: Map[String, Any] = Map(
    "numTrees" -> 1,
    "maxSamples" -> 1.0,
    "maxFeatures" -> 1.0,
    "maxDepth" -> 3,
    "contamination" -> 0.2,
    "bootstrap" -> false
  )
}
