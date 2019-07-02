package org.apache.spark.ml.iforest


import scala.collection.mutable
import scala.util.Random
import org.apache.commons.math3.random.{RandomDataGenerator, RandomGeneratorFactory}
import org.apache.hadoop.fs.Path
import org.apache.log4j.Logger
import org.apache.spark.SparkException
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructType}

import scala.reflect.ClassTag


/**
  * Model of IF(isolation forest), including constructor, copy, write and get summary.
  * @param uid unique ID for the Model
  * @param _trees Param of trees for constructor
  */
class IForestModel (
    override val uid: String,
    private val _trees: Array[IFNode]) extends Model[IForestModel]
    with IForestParams with MLWritable {

  require(_trees.nonEmpty, "IForestModel requires at least 1 tree.")

  import IForestModel._

  def trees: Array[IFNode] = _trees

  /** add extra param to the Model */
  override def copy(extra: ParamMap): IForestModel = {
    val copied = copyValues(new IForestModel(uid, trees), extra)
    copied.setSummary(trainingSummary).setParent(parent)
  }

  override def write: MLWriter = new IForestModel.IForestModelWriter(this)

  private var trainingSummary: Option[IForestSummary] = None

  // Threshold for anomaly score. Default is -1.
  private var threshold: Double = -1d

  private[iforest] def setSummary(summary: Option[IForestSummary]): this.type = {
    this.trainingSummary = summary
    this
  }

  /**
    * Return true if there exists summary of model
    */
  def hasSummary: Boolean = trainingSummary.nonEmpty

  def summary: IForestSummary = trainingSummary.getOrElse {
    throw new SparkException(
      s"No training summary available for the ${this.getClass.getSimpleName}"
    )
  }

  def getThreshold(): Double = {
    this.threshold
  }

  def setThreshold(value: Double): this.type = {
    this.threshold = value
    this
  }

  /**
    * Predict if a particular sample is an outlier or not.
    * @param dataset Input data which is a dataset with n_samples rows. This dataset must have a
    *                column named features, or call setFeaturesCol to set user defined feature
    *                column name. This column stores the feature values for each instance, users can
    *                use VectorAssembler to generate a feature column.
    * @return A predicted dataframe with a prediction column which stores prediction values.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val numSamples = dataset.count()
    val possibleMaxSamples =
      if ($(maxSamples) > 1.0) $(maxSamples) else ($(maxSamples) * numSamples)
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    // calculate anomaly score
    val scoreUDF = udf { (features: Vector) => {
      val normFactor = avgLength(possibleMaxSamples)
      val avgPathLength = bcastModel.value.calAvgPathLength(features)
      Math.pow(2, -avgPathLength / normFactor)
    }
    }
    // append a score column
    val scoreDataset = dataset.withColumn($(anomalyScoreCol), scoreUDF(col($(featuresCol))))

    if (threshold < 0) {
      logger.info("threshold is not set, calculating the anomaly threshold according to param contamination..")
      threshold = scoreDataset.stat.approxQuantile($(anomalyScoreCol),
        Array(1 - $(contamination)), $(approxQuantileRelativeError))(0)
    }

    // set anomaly instance label 1
    val predictUDF = udf { (anomalyScore: Double) =>
      if (anomalyScore > threshold) 1.0 else 0.0
    }
    scoreDataset.withColumn($(predictionCol), predictUDF(col($(anomalyScoreCol))))
  }

  /**
    * Calculate an average path length for a given feature set in a forest.
    * @param features A Vector stores feature values.
    * @return Average path length.
    */
  private def calAvgPathLength(features: Vector): Double = {
    val avgPathLength = trees.map(ifNode => {
      calPathLength(features, ifNode, 0)
    }).sum / trees.length
    avgPathLength
  }

  /**
    * Calculate a path langth for a given feature set in a tree.
    * @param features A Vector stores feature values.
    * @param ifNode Tree's root node.
    * @param currentPathLength Current path length.
    * @return Path length in this tree.
    */
  private def calPathLength(features: Vector,
      ifNode: IFNode,
      currentPathLength: Int): Double = ifNode match {
    case leafNode: IFLeafNode => currentPathLength + avgLength(leafNode.numInstance)
    case internalNode: IFInternalNode =>
      val attrIndex = internalNode.featureIndex
      if (features(attrIndex) < internalNode.featureValue) {
        calPathLength(features, internalNode.leftChild, currentPathLength + 1)
      } else {
        calPathLength(features, internalNode.rightChild, currentPathLength + 1)
      }
  }

  /**
    * A function to calculate an expected path length with a specific data size.
    * @param size Data size.
    * @return An expected path length.
    */
  private def avgLength(size: Double): Double = {
    if (size > 2) {
      val H = Math.log(size - 1) + EulerConstant
      2 * H - 2 * (size - 1) / size
    }
    else if (size == 2) 1.0
    else 0.0
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}


/**
  * object of isolation forest Model
  */
object IForestModel extends MLReadable[IForestModel] {

  val EulerConstant = 0.5772156649

  private val logger = Logger.getLogger(IForestModel.getClass)

  override def read: MLReader[IForestModel] = new IForestModelReader

  override def load(path: String): IForestModel = super.load(path)

  /**
    * Info for a IFNode
    *
    * @param id Index used for tree reconstruction. Indices follow a pre-order traversal
    * @param featureIndex Feature index stored in a internal node, or -1 if leaf node
    * @param featureValue Feature value stored in a internal node, or -1 if leaf node
    * @param leftChild Left child index, or -1 if leaf node
    * @param rightChild Right child index, or -1 if leaf node
    * @param numInstance The number of instances in a leaf node, or 0 if internal node
    */
  private case class NodeData(
      id: Int,
      featureIndex: Int,
      featureValue: Double,
      leftChild: Int,
      rightChild: Int,
      numInstance: Long
  )

  private object NodeData {
    /**
      *
      * @param node IFNode instance
      * @param id Current ID, IDs are assigned via a pre-order traversal
      * @return (sequence of nodes in pre-order traversal order, largest ID in subtree)
      *         The nodes are returned in pre-order traversal (root first) so that it is
      *         easy to get the ID of the subtree's root node.
      */
    def build(node: IFNode, id: Int): (Seq[NodeData], Int) = node match {
      case n: IFInternalNode =>
        val (leftNodeData, leftIdx) = build(n.leftChild, id + 1)
        val (rightNodeData, rightIdx) = build(n.rightChild, leftIdx + 1)
        val thisNodeData = NodeData(id, n.featureIndex, n.featureValue,
          leftNodeData.head.id, rightNodeData.head.id, 0)
        (thisNodeData +: (leftNodeData ++ rightNodeData), rightIdx)
      case n: IFLeafNode =>
        (Seq(NodeData(id, -1, -1.0, -1, -1, n.numInstance)), id)
    }
  }

  /**
    * Info for a forest
    *
    * @param treeID tree index in a forest
    * @param nodeData info for a IFNode
    */
  private case class EnsembleNodeData (
      treeID: Int,
      nodeData: NodeData
  )

  /**
    * load iforest tree nodes from a file path
    * @param path load path
    * @param sparkSession spark session
    * @return array of root node
    */
  private def loadTreeNodes(
      path: String,
      sparkSession: SparkSession): Array[IFNode] = {

    import sparkSession.implicits._
    val dataPath = new Path(path, "data").toString
    val nodeData: Dataset[EnsembleNodeData] =
      sparkSession.read.parquet(dataPath).as[EnsembleNodeData]
    val rootNodesRDD: RDD[(Int, IFNode)] =
      nodeData.rdd.map(d => (d.treeID, d.nodeData)).groupByKey().map {
        case (treeID: Int, nodeData: Iterable[NodeData]) =>
          treeID -> buildFromNodes(nodeData.toArray)
      }
    val rootNodes: Array[IFNode] = rootNodesRDD.sortByKey().values.collect()
    rootNodes
  }

  /**
    * reconstruct a tree from nodes array
    *
    * @param data node data array
    * @return tree's root node
    */
  private def buildFromNodes(data: Array[NodeData]): IFNode = {
    // Load all nodes, sorted by ID.
    val nodes = data.sortBy(_.id)
    require(nodes.head.id == 0, s"Tree load failed. Expected smallet node ID to be," +
        s" but found ${nodes.head.id}")
    require(nodes.last.id == nodes.length - 1, s"Tree load failed, Expected largest node " +
        s"ID to be, but found ${nodes.last.id}")

    val finalNodes = new Array[IFNode](nodes.length)
    nodes.reverseIterator.foreach {   case n: NodeData =>
      val node = if (n.leftChild != -1) {
        val leftChild = finalNodes(n.leftChild)
        val rightChild = finalNodes(n.rightChild)
        new IFInternalNode(leftChild, rightChild, n.featureIndex, n.featureValue)
      }
      else {
        new IFLeafNode(n.numInstance)
      }
      finalNodes(n.id) = node
    }
    // Retrun the root node
    finalNodes.head
  }

  private[IForestModel] class IForestModelWriter(instance: IForestModel) extends MLWriter {
    override protected def saveImpl(path: String) {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: tree nodes
      val dataRDD = sc.parallelize(instance.trees.zipWithIndex).flatMap(
        elem => {
          val (nodeData: Seq[NodeData], _) = NodeData.build(elem._1, 0)
          nodeData.map(nd => EnsembleNodeData(elem._2, nd))}
      )
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(dataRDD).write.parquet(dataPath)
    }
  }

  private class IForestModelReader extends MLReader[IForestModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[IForestModel].getName

    override def load(path: String): IForestModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val trees = loadTreeNodes(path, sparkSession)
      val model = new IForestModel(metadata.uid, trees)
      metadata.getAndSetParams(model)
      model
    }
  }
}

/**
  * Isolation Forest (iForest) is a effective model that focuses on anomaly isolation.
  * iForest uses tree structure for modeling data, iTree isolates anomalies closer to
  * the root of the tree as compared to normal points.
  *
  * A anomaly score is calculated by iForest model to measure the abnormality of the
  * data instances. The higher, the more abnormal.
  *
  * More details about iForest can be found in paper
  * <a href="https://dl.acm.org/citation.cfm?id=1511387">Isolation Forest</a>
  *
  * iForest on Spark is trained via model-wise parallelism, and predicts a new Dataset via data-wise parallelism,
  * It is implemented in the following steps:
  * 1. Sampling data from a Dataset. Data instances are sampled and grouped for each iTree. As indicated in the paper,
  * the number samples for constructing each tree is usually not very large (default value 256). Thus we can construct
  * a sampled paired RDD, where each row key is tree index and row value is a group of sampled data instances for a tree.
  * 2. Training and constructing each iTree on parallel via a map operation and collect the iForest model in the driver.
  * 3. Predict a new Dataset on parallel via a map operation with the collected iForest model.
  *
  * @param uid unique ID for Model
  */
class IForest (
    override val uid: String) extends Estimator[IForestModel]
    with IForestParams with DefaultParamsWritable {

  import IForest._

  setDefault(
    numTrees -> 100,
    maxSamples -> 1.0,
    maxFeatures -> 1.0,
    maxDepth -> 10,
    contamination -> 0.1,
    bootstrap -> false,
    seed -> this.getClass.getName.hashCode.toLong,
    approxQuantileRelativeError -> 0d
  )

  def this() = this(Identifiable.randomUID("IForest"))


  /** @group setParam */
  def setNumTrees(value: Int): this.type = set(numTrees, value)

  /** @group setParam */
  def setMaxSamples(value: Double): this.type = set(maxSamples, value)

  /** @group setParam */
  def setMaxFeatures(value: Double): this.type = set(maxFeatures, value)

  /** @group setParam */
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  def setContamination(value: Double): this.type = set(contamination, value)

  /** @group setParam */
  def setBootstrap(value: Boolean): this.type = set(bootstrap, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /** @group setParam */
  def setAnomalyScoreCol(value: String): this.type = set(anomalyScoreCol, value)

  /** @group setParam */
  def setApproxQuantileRelativeError(value: Double): this.type = set(approxQuantileRelativeError, value)

  override def copy(extra: ParamMap): IForest = defaultCopy(extra)

  lazy val rng = new Random($(seed))

  var numSamples = 0L
  var possibleMaxSampels = 0

  /**
    * Sample and split data to $numTrees groups, each group will build a tree.
    *
    *
    * @param dataset Training Dataset
    * @return A paired RDD, where key is the tree index, value is an array of data instances for training a iTree.
    */
  private[iforest] def splitData(dataset: Dataset[_]): RDD[(Int, Array[Vector])] = {
    numSamples = dataset.count()
    val fraction =
      if ($(maxSamples) > 1) $(maxSamples) / numSamples
      else $(maxSamples)

    require(fraction <= 1.0, "The max samples must be less then total number of the input data")

    possibleMaxSampels = (fraction * numSamples).toInt

    // use advanced apache common math3 random generator
    val advancedRgn = new RandomDataGenerator(
      RandomGeneratorFactory.createRandomGenerator(new java.util.Random(rng.nextLong()))
    )


    val rddPerTree = {

      // SampledIndices is a two-dimensional array, that generates sampling row indices in each iTree.
      // E.g. [[1, 3, 6, 4], [6, 4, 2, 5]] indicates that the first tree has
      // data consists of the 1, 3, 6, 4 row samples, the second tree has data
      // consists of the 6, 4, 3, 5 row samples.
      // if bootstrap is true, each array can stores the repeated row indices
      // if false, each array contains different row indices, and each index is
      // elected with the same probability using reservoir sample method.
      // Note: sampleIndices will occupy about maxSamples * numTrees * 8
      // byte memory in the driver.
      val sampleIndices = if ($(bootstrap)) {
        Array.tabulate($(numTrees)) { i =>
          Array.fill(possibleMaxSampels) {
            advancedRgn.nextLong(0, numSamples)
          }
        }
      } else {
        Array.tabulate($(numTrees)) { i =>
          reservoirSampleAndCount(Range.Long(0, numSamples, 1).iterator,
            possibleMaxSampels, rng.nextLong)._1
        }
      }

      // rowInfo structure is a Map in which key is rowId identifying each data instance,
      // and value is a SparseVector that indicating this data instance is sampled for training which iTrees.
      // SparseVector is constructed by (numTrees, treeIdArray, numCopiesArray), where
      //  - treeIdArray indicates that which tree this row data is trained on;
      //  - numCopiesArray indicates how namy copies of this row data in the corresponding tree.
      //
      // E.g., Map{1 -> SparseVector(100, [1, 3, 5], [3, 6, 1])} means that there are 100
      // trees to construct a forest, and 3 copies of 1st row data trained on the 1 tree,
      // 6 copies trained on the 3rd tree and 1 copy trained on the 5th tree.
      val rowInfo = sampleIndices.zipWithIndex.flatMap { case (indices: Array[Long], treeId: Int) =>
        indices.map(rowIndex => (rowIndex, treeId))
      }.groupBy(_._1).mapValues(x => x.map(_._2)).map {
        case (rowIndex: Long, treeIdArray: Array[Int]) =>
          val treeIdWithNumCopies = treeIdArray.map(treeId => (treeId, 1.0))
              .groupBy(_._1).map { case (treeId: Int, tmp: Array[Tuple2[Int, Double]]) =>
            tmp.reduce((x, y) => (treeId, x._2 + y._2))
          }.toSeq
          (rowIndex, Vectors.sparse($(numTrees), treeIdWithNumCopies))
      }

      val broadRowInfo = dataset.sparkSession.sparkContext.broadcast(rowInfo)

      // Firstly filter rows that contained in the rowInfo, i.e., the instances
      // that are used to construct the forest.
      // Then for each row, get the number of copies in each tree, copy this point
      // to an array with corresponding tree id.
      // Finally reduce by the tree id key.
      dataset.select(col($(featuresCol))).rdd.map {
        case Row(point: Vector) => point
      }.zipWithIndex().filter{ case (point: Vector, rowIndex: Long) =>
        broadRowInfo.value.contains(rowIndex)
      }.flatMap { case (point: Vector, rowIndex: Long) =>
        val numCopiesInEachTree = broadRowInfo.value.get(rowIndex).get.asInstanceOf[SparseVector]
        numCopiesInEachTree.indices.zip(numCopiesInEachTree.values).map {
          case (treeId: Int, numCopies: Double) =>
            (treeId, Array.fill(numCopies.toInt)(point))
        }
      }.reduceByKey((arr1, arr2) => arr1 ++ arr2)
    }
    rddPerTree
  }

  /**
    * Training a iforest model for a given dataset
    *
    * @param dataset Input data which is a dataset with n_samples rows. This dataset must have a
    *                column named features, or call setFeaturesCol to set user defined feature
    *                column name. This column stores the feature values for each instance, users can
    *                use VectorAssembler to generate a feature column.
    * @return trained iforest model with an array of each tree's root node.
    */
  override def fit(dataset: Dataset[_]): IForestModel = instrumented { instr =>
    transformSchema(dataset.schema, logging = true)

    val rddPerTree = splitData(dataset)

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, numTrees, maxSamples, maxFeatures, maxDepth, contamination,
      approxQuantileRelativeError, bootstrap, seed, featuresCol, predictionCol, labelCol)

    // Each iTree of the iForest will be built on parallel and collected in the driver.
    // Approximate memory usage for iForest model is calculated, a warning will be raised if iForest is too large.
    val usageMemery = $(numTrees) * 2 * possibleMaxSampels * 32 / (1024 * 1024)
    if (usageMemery > 256) {
      logger.warn("The isolation forest stored on the driver will exceed 256M memory. " +
          "If your machine can not bear memory consuming, please try small numTrees or maxSamples.")
    }

    // build each tree and construct a forest
    val _trees = rddPerTree.map {
      case (treeId: Int, points: Array[Vector]) =>
        // Create a random for iTree generation
        val random = new Random(rng.nextInt + treeId)

        // sample features
        val (trainData, featureIdxArr) = sampleFeatures(points, $(maxFeatures), random)

        // calculate actual maxDepth to limit tree height
        val longestPath = Math.ceil(Math.log(Math.max(2, points.length)) / Math.log(2)).toInt
        val possibleMaxDepth = if ($(maxDepth) > longestPath) longestPath else $(maxDepth)
        if(possibleMaxDepth != $(maxDepth)) {
          logger.warn("building itree using possible max depth " + possibleMaxDepth + ", instead of " + $(maxDepth))
        }

        val numFeatures = trainData.head.size
        // a array stores constant features index
        val constantFeatures = Array.tabulate(numFeatures + 1) {
          i => i
        }
        // last position's value indicates constant feature offset index
        constantFeatures(numFeatures) = 0
        // build a tree
        iTree(trainData, 0, possibleMaxDepth, constantFeatures, featureIdxArr, random)

    }.collect()

    val model = copyValues(new IForestModel(uid, _trees).setParent(this))
    val predictions = model.transform(dataset)
    val summary = new IForestSummary(
      predictions, $(featuresCol), $(predictionCol), $(anomalyScoreCol)
    )

    model.setSummary(Some(summary))
    model
  }

  /**
    * Sample features to train a tree.
    * @param data Input data to train a tree, each element is an instance.
    * @param maxFeatures The number of features to draw.
    * @return Tuple (sampledFeaturesDataset, featureIdxArr),
    *         featureIdxArr is an array stores the origin feature idx before the feature sampling
    */
  private[iforest] def sampleFeatures(
      data: Array[Vector],
      maxFeatures: Double,
      random: Random = new Random()): (Array[Array[Double]], Array[Int]) = {

    // get feature size
    val numFeatures = data.head.size
    // calculate the number of sampling features
    val subFeatures: Int =
      if (maxFeatures <= 1) (maxFeatures * numFeatures).toInt
      else if (maxFeatures > numFeatures) {
        logger.warn("maxFeatures is larger than the numFeatures, using all features instead")
        numFeatures
      }
      else maxFeatures.toInt

    if (subFeatures == numFeatures) {
      (data.toArray.map(vector => vector.asInstanceOf[DenseVector].values), Array.range(0, numFeatures))
    } else {
      // feature index for sampling features
      val featureIdx = random.shuffle(0 to numFeatures - 1).take(subFeatures)

      val sampledFeatures = mutable.ArrayBuilder.make[Array[Double]]
      data.foreach(vector => {
        val sampledValues = new Array[Double](subFeatures)
        featureIdx.zipWithIndex.foreach(elem => sampledValues(elem._2) = vector(elem._1))
        sampledFeatures += sampledValues
      })
      (sampledFeatures.result(), featureIdx.toArray)
    }
  }

  /**
    * Builds a tree
    *
    * @param data Input data, a two dimensional array, can be regarded as a table, each row
    *             is an instance, each column is a feature value.
    * @param currentPathLength current node's path length
    * @param maxDepth height limit during building a tree
    * @param constantFeatures an array stores constant features indices, constant features
    *                         will not be drawn
    * @param featureIdxArr an array stores the mapping from the sampled feature idx to the origin feature idx
    * @param randomSeed random for generating iTree
    * @return tree's root node
    */
  private[iforest] def iTree(data: Array[Array[Double]],
      currentPathLength: Int,
      maxDepth: Int,
      constantFeatures: Array[Int],
      featureIdxArr: Array[Int],
      random: Random): IFNode = {

    var constantFeatureIndex = constantFeatures.last
    // the condition of leaf node
    // 1. current path length exceeds max depth
    // 2. the number of data can not be splitted again
    // 3. there are no non-constant features to draw
    if (currentPathLength >= maxDepth || data.length <= 1) {
      new IFLeafNode(data.length)
    } else {
      val numFeatures = data.head.length
      var attrMin = 0.0
      var attrMax = 0.0
      var attrIndex = -1
      // until find a non-constant feature
      var findConstant = true
      while (findConstant && numFeatures != constantFeatureIndex) {
        // select randomly a feature index
        val idx = random.nextInt(numFeatures - constantFeatureIndex) + constantFeatureIndex
        attrIndex = constantFeatures(idx)
        val features = Array.tabulate(data.length)( i => data(i)(attrIndex))
        attrMin = features.min
        attrMax = features.max
        if (attrMin == attrMax) {
          // swap constant feature index with non-constant feature index
          constantFeatures(idx) = constantFeatures(constantFeatureIndex)
          constantFeatures(constantFeatureIndex) = attrIndex
          // constant feature index add 1, then update
          constantFeatureIndex += 1
          constantFeatures(constantFeatures.length - 1) = constantFeatureIndex
        } else {
          findConstant = false
        }
      }
      if (numFeatures == constantFeatureIndex) new IFLeafNode(data.length)
      else {
        // select randomly a feature value between (attrMin, attrMax)
        val attrValue = random.nextDouble() * (attrMax - attrMin) + attrMin
        // split data according to the attrValue
        val leftData = data.filter(point => point(attrIndex) < attrValue)
        val rightData = data.filter(point => point(attrIndex) >= attrValue)
        // recursively build a tree
        new IFInternalNode(
          iTree(leftData, currentPathLength + 1, maxDepth, constantFeatures.clone(), featureIdxArr, random),
          iTree(rightData, currentPathLength + 1, maxDepth, constantFeatures.clone(), featureIdxArr, random),
          featureIdxArr(attrIndex), attrValue)
      }
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

}

object IForest extends DefaultParamsReadable[IForest] {

  val logger = Logger.getLogger(IForest.getClass)

  override def load(path: String): IForest = super.load(path)
}

trait IForestParams extends Params {
  /**
    * The number of trees in the iforest model (>0).
    * @group param
    */
  final val numTrees: IntParam =
    new IntParam(this, "numTrees", "the number of trees in the iForest. " +
        "Must be > 0", ParamValidators.gt(0))

  /** @group getParam */
  def getNumTrees: Int = $(numTrees)

  /**
    * The number of samples to draw from data to train each tree (>0).
    *
    * If <= 1, the algorithm will draw maxSamples * totalSample samples.
    *
    * If > 1, the algorithm will draw maxSamples samples.
    *
    * This parameter will affect the driver's memory when splitting data.
    *
    * The total memory is about maxSamples * numTrees * 4 + maxSamples * 8 bytes.
    *
    * @group param
    */
  final val maxSamples: DoubleParam =
    new DoubleParam(this, "maxSamples", "the number of samples to " +
        "draw from data to train each tree. Must be > 0. If <= 1, " +
        "then draw maxSamples * totalSamples. If > 1, then draw " +
        "maxSamples samples.", ParamValidators.gt(0.0))

  /** @group getParam */
  def getMaxSamples: Double = $(maxSamples)

  /**
    * The number of features to draw from data to train each tree (>0).
    *
    * If <= 1, the algorithm will draw maxFeatures * totalFeatures features.
    *
    * If > 1, the algorithm will draw maxFeatures features.
    * @group param
    */
  final val maxFeatures: DoubleParam =
    new DoubleParam(this, "maxFeatures", "the number of features to" +
        " draw from data to train each tree. Must be > 0. If <= 1, " +
        "then draw maxFeatures * totalFeatures. If > 1, then draw " +
        "maxFeatures features.", ParamValidators.gt(0.0))

  /** @group getParam */
  def getMaxFeatures: Double = $(maxFeatures)

  /**
    * The height limit used in constructing a tree (>0).
    *
    * The default value will be about log2(numSamples).
    * @group param
    */
  final val maxDepth: IntParam =
    new IntParam(this, "maxDepth", "the height limit used in constructing" +
        " a tree. Must be > 0", ParamValidators.gt(0))

  /** @group getParam */
  def getMaxDepth: Int = $(maxDepth)

  /**
    * The proportion of outliers in the data set (0< contamination < 1).
    * It will be used in the prediction. In order to enhance performance,
    * Our method to get anomaly score threshold adopts DataFrameStsFunctions.approxQuantile,
    * which is designed for performance with some extent accuracy loss.
    * Set the param approxQuantileRelativeError (0 < e < 1) to calculate
    * an approximate quantile threshold of anomaly scores for large dataset.
    * @group param
    */
  final val contamination: DoubleParam =
    new DoubleParam(this, "contamination", "the proportion of " +
        "outliers in the data set. Must be > 0", ParamValidators.inRange(0, 1, false, true))

  /** @group getParam */
  def getContamination: Double = $(contamination)

  /**
    * Relative Error for Approximate Quantile (0 <= value <= 1),  default is 0.
    * @group param
    */
  final val approxQuantileRelativeError: DoubleParam =
    new DoubleParam(parent = this, name ="approxQuantileRelativeError", doc = "relative error for approximate quantile")

  /** @group setParam */
  setDefault(approxQuantileRelativeError, value = 0d)

  /** @group getParam */
  final def getApproxQuantileRelativeError: Double = $(approxQuantileRelativeError)

  /**
    * If true, individual trees are fit on random subsets of the training data
    * sampled with replacement. If false, sampling without replacement is performed.
    * @group param
    */
  final val bootstrap: BooleanParam =
    new BooleanParam(this, "bootstrap", "If false, samples in a tree " +
        "are not the same, i.e. draw without replacement. If true, samples in a tree" +
        " are drawn with replacement.")

  /** @group getParam */
  def getBootstrap: Boolean = $(bootstrap)

  /**
    * The seed used by the random number generator.
    * @group param
    */
  final val seed: LongParam = new LongParam(this, "seed", "random seed")

  /** @group getParam */
  def getSeed: Long = $(seed)

  /**
    * features column name, used in the dataset.
    * @group param
    */
  final val featuresCol: Param[String] =
    new Param[String](this, "featuresCol", "features column name")

  /** @group setParam */
  setDefault(featuresCol, "features")

  /** @group getParam */
  final def getFeaturesCol: String = $(featuresCol)

  /**
    * label column name, used in the dataset.
    *
    * It's only used in testing the algorithm's performance.
    * @group param
    */
  final val labelCol: Param[String] =
    new Param[String](this, "labelCol", "label column name")

  /** @group setParam */
  setDefault(labelCol, "label")

  /** @group getParam */
  final def getLabelCol: String = $(labelCol)

  /**
    * Prediction column name, used in the dataset.
    * @group param
    */
  final val predictionCol: Param[String] =
    new Param[String](this, "predictionCol", "prediction column name")

  /** @group setParam */
  setDefault(predictionCol, "prediction")

  /** @group getParam */
  final def getPredictionCol: String = $(predictionCol)

  /**
    * Anomaly score column name, used in the dataset.
    * @group param
    */
  final val anomalyScoreCol: Param[String] =
    new Param[String](this, "anomalyScoreCol", "anomaly score column name")

  /** @group setParam */
  setDefault(anomalyScoreCol, "anomalyScore")

  /** @group getParam */
  final def getAnomalyScoreCol: String = $(anomalyScoreCol)

  /**
    * Validates and transforms the input schema.
    * @param schema input schema
    * @return output schema
    */
  def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT, "")
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  /**
    * Reservoir sampling implementation that also returns the input size.
    * @param input input size
    * @param k reservoir size
    * @param seed random seed
    * @return (samples, input size)
    */
  def reservoirSampleAndCount[T: ClassTag](
      input: Iterator[T],
      k: Int,
      seed: Long = Random.nextLong()): (Array[T], Long) = {
    val reservoir = new Array[T](k)
    // Put the first k elements in the reservoir.
    var i = 0
    while (i < k && input.hasNext) {
      val item = input.next()
      reservoir(i) = item
      i += 1
    }

    // If we have consumed all the elements, return them. Otherwise do the replacement.
    if (i < k) {
      // If input size < k, trim the array to return only an array of input size
      val trimReservoir = new Array[T](i)
      System.arraycopy(reservoir, 0, trimReservoir, 0, i)
      (trimReservoir, i)
    } else {
      // If input size > k, continue the sampling process.
      var l = i.toLong
      val rand = new Random(seed)
      while (input.hasNext) {
        val item = input.next()
        l += 1
        // There are k elements in the reservoir, and the l-th element has been
        // consumed. It should be chosen with probability k/l. The expression
        // below is a random long chosen uniformly from [0,l)
        val replacementIndex = (rand.nextDouble() * l).toLong
        if (replacementIndex < k) {
          reservoir(replacementIndex.toInt) = item
        }
      }
      (reservoir, l)
    }
  }
}

class IForestSummary (
    @transient val predictions: DataFrame,
    val featuresCol: String,
    val predictionCol: String,
    val anomalyScoreCol: String
) extends Serializable {

  @transient lazy val anomalies: DataFrame = predictions.select(predictionCol)

  @transient lazy val anomalyScores: DataFrame = predictions.select(anomalyScoreCol)

  def numAnomalies: Long = anomalies.where(col(predictionCol) > 0).collect().length
}


