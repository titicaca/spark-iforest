from pyspark import since, keyword_only
from pyspark.ml.util import *
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaParams, JavaWrapper
from pyspark.ml.param.shared import *
from pyspark.ml.common import inherit_doc
from pyspark_iforest.ml.util import *

__all__ = ['IForestSummary', 'IForest', 'IForestModel']


class IForestSummary(JavaWrapper):
    """
    .. note:: Experimental

    Clustering results for IForest model.

    .. versionadded:: 2.1.0
    """

    @property
    @since("2.1.0")
    def predictionCol(self):
        """
        Name for column of predicted clusters in `predictions`.
        """
        return self._call_java("predictionCol")

    @property
    @since("2.1.0")
    def predictions(self):
        """
        DataFrame produced by the model's `transform` method.
        """
        return self._call_java("predictions")

    @property
    @since("2.1.0")
    def featuresCol(self):
        """
        Name for column of features in `predictions`.
        """
        return self._call_java("featuresCol")

    @property
    @since("2.1.0")
    def anomalyScoreCol(self):
        """
        Name for column of anomalyScore.
        """
        return self._call_java("anomalyScoreCol")

    @property
    @since("2.1.0")
    def anomalies(self):
        """
        DataFrame of predicted anomalies for each training data point.
        """
        return self._call_java("anomalies")

    @property
    @since("2.1.0")
    def anomalyScores(self):
        """
        DataFrame of predicted anomalyScores for each training data point.
        """
        return self._call_java("anomalyScores")

    @property
    @since("2.1.0")
    def numAnomalies(self):
        """
        Number of anomalies.
        """
        return self._call_java("numAnomalies")


class CustomizedJavaMLReader(JavaMLReader):
    @classmethod
    def _java_loader_class(cls, clazz):
        """
        Returns the full class name of the Java ML instance. The default
        implementation replaces "pyspark" by "org.apache.spark" in
        the Python full class name.
        """
        java_package = clazz.__module__.replace("pyspark_iforest", "org.apache.spark")
        if clazz.__name__ in ("Pipeline", "PipelineModel"):
            # Remove the last package name "pipeline" for Pipeline and PipelineModel.
            java_package = ".".join(java_package.split(".")[0:-1])
        return java_package + "." + clazz.__name__

    def load(self, path):
        """Load the ML instance from the input path."""
        if not isinstance(path, basestring):
            raise TypeError("path should be a basestring, got type %s" % type(path))
        java_obj = self._jread.load(path)
        if not hasattr(self._clazz, "_from_java"):
            raise NotImplementedError("This Java ML type cannot be loaded into Python currently: %r"
                                      % self._clazz)
        return customized_from_java(java_obj)


@inherit_doc
class CustomizedJavaMLReadable(MLReadable):
    """
    (Private) Mixin for instances that provide JavaMLReader.
    """

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return CustomizedJavaMLReader(cls)


class IForestModel(JavaModel, JavaMLWritable, CustomizedJavaMLReadable):
    """
    Model fitted by IForest.
    .. versionadded:: 2.1.0
    """

    @property
    @since("2.1.0")
    def hasSummary(self):
        """
        Indicates whether a training summary exists for this model instance.
        """
        return self._call_java("hasSummary")

    @property
    @since("2.1.0")
    def summary(self):
        """
        Gets summary of the model trained on the training set.
        An exception is thrown if no summary exists.
        """
        if self.hasSummary:
            return IForestSummary(self._call_java("summary"))
        else:
            raise RuntimeError("No training summary available for this %s" %
                               self.__class__.__name__)

    @since("2.4.0")
    def setThreshold(self, value):
        """
        Set the anomaly score threshold for prediction, 0 < value < 1 .
        """
        self._call_java("setThreshold", value)

    @since("2.4.0")
    def getThreshold(self):
        """
        Get the anomaly score threshold for prediction, 0 < value < 1 .
        """
        return self._call_java("getThreshold")

@inherit_doc
class IForest(JavaEstimator, HasFeaturesCol, HasPredictionCol, HasSeed, JavaMLWritable, CustomizedJavaMLReadable):
    """
    Isolation Forest for detecting anomalies

    >>> from pyspark.ml.linalg import Vectors
    >>>
    >>> data = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([7.0, 9.0]),),
    ...             (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)]
    >>>
    ...
    ... df = spark.createDataFrame(data, ["features"])
    >>>
    >>>
    >>> from pyspark_iforest.ml.iforest import *
    >>>
    >>> iforest = IForest(contamination=0.3, maxDepth=2)
    >>> model = iforest.fit(df)
    >>>
    >>> model.hasSummary
    True
    >>>
    >>> summary = model.summary
    >>>
    >>> summary.numAnomalies
    1
    >>>
    >>> transformed = model.transform(df)
    >>>
    >>> rows = transformed.collect()
    >>>
    >>> import tempfile
    >>> temp_path = tempfile.mkdtemp()
    >>>
    >>> iforest_path = temp_path + "/iforest"
    >>>
    >>> iforest.save(iforest_path)
    >>>
    >>> loaded_iforest = IForest.load(iforest_path)
    >>>
    >>> model_path = temp_path + "/iforest_model"
    >>>
    >>> model.save(model_path)
    >>>
    >>> loaded_model = IForestModel.load(model_path)
    >>>
    >>> loaded_model.hasSummary
    False
    >>>
    >>> loaded_model.transform(df).show()
    +---------+-------------------+----------+
    | features|       anomalyScore|prediction|
    +---------+-------------------+----------+
    |[0.0,0.0]|  0.652628934546283|       1.0|
    |[7.0,9.0]| 0.3806804982830844|       0.0|
    |[9.0,8.0]|0.40116303198069875|       0.0|
    |[8.0,9.0]|  0.366693565357915|       0.0|
    +---------+-------------------+----------+

    .. versionadded:: 2.1.0
    """

    numTrees = Param(Params._dummy(), "numTrees", "The number of trees to create. Must be > 1.",
                     typeConverter=TypeConverters.toInt)
    maxSamples = Param(Params._dummy(), "maxSamples",
                       "The number of samples to draw from data to train each tree (>0)",
                       typeConverter=TypeConverters.toFloat)
    maxFeatures = Param(Params._dummy(), "maxFeatures",
                        "The number of features to draw from data to train each tree (>0)",
                        typeConverter=TypeConverters.toFloat)
    maxDepth = Param(Params._dummy(), "maxDepth",
                     "The height limit used in constructing a tree (>0)",
                     typeConverter=TypeConverters.toInt)
    contamination = Param(Params._dummy(), "contamination",
                          "The proportion of outliers in the data set (0< contamination < 1)",
                          typeConverter=TypeConverters.toFloat)
    bootstrap = Param(Params._dummy(), "bootstrap",
                      "If true, the training data sampled with replacement (boolean)",
                      typeConverter=TypeConverters.toBoolean)

    approxQuantileRelativeError = Param(Params._dummy(), "approxQuantileRelativeError",
                                        "Relative Error for anomaly score approximate quantile calculaion (0 <= e <= 1)",
                                        typeConverter=TypeConverters.toFloat)

    @keyword_only
    def __init__(self, featuresCol="features", predictionCol="prediction", anomalyScore="anomalyScore",
                 numTrees=100, maxSamples=1.0, maxFeatures=1.0, maxDepth=10, contamination=0.1,
                 bootstrap=False, approxQuantileRelativeError=0.):

        super(IForest, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.iforest.IForest", self.uid)
        self._setDefault(numTrees=100, maxSamples=1.0, maxFeatures=1.0, maxDepth=10, contamination=0.1,
                         bootstrap=False, approxQuantileRelativeError=0.)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def _create_model(self, java_model):
        return IForestModel(java_model)

    @keyword_only
    @since("2.1.0")
    def setParams(self, featuresCol="features", predictionCol="prediction", anomalyScore="anomalyScore",
                  numTrees=100, maxSamples=1.0, maxFeatures=1.0, maxDepth=10, contamination=0.1,
                  bootstrap=False, approxQuantileRelativeError=0.):
        """
        Sets params for IForest.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @since("2.1.0")
    def setNumTrees(self, value):
        """
        Sets the value of :py:attr:`numTrees`.
        """
        return self._set(numTrees=value)

    @since("2.1.0")
    def getNumTrees(self):
        """
        Gets the value of `numTrees`
        """
        return self.getOrDefault(self.numTrees)

    @since("2.1.0")
    def setMaxSamples(self, value):
        """
        Sets the value of :py:attr:`maxSamples`.
        """
        return self._set(maxSamples=value)

    @since("2.1.0")
    def getMaxSamples(self):
        """
        Gets the value of `maxSamples`
        """
        return self.getOrDefault(self.maxSamples)

    @since("2.1.0")
    def setMaxFeatures(self, value):
        """
        Sets the value of :py:attr:`maxFeatures`.
        """
        return self._set(maxFeatures=value)

    @since("2.1.0")
    def getMaxFeatures(self):
        """
        Gets the value of `maxFeatures`
        """
        return self.getOrDefault(self.MaxSamples)

    @since("2.1.0")
    def setMaxDepth(self, value):
        """
        Sets the value of :py:attr:`maxDepth`.
        """
        return self._set(maxDepth=value)

    @since("2.1.0")
    def getMaxDepth(self):
        """
        Gets the value of `maxDepth`
        """
        return self.getOrDefault(self.maxDepth)

    @since("2.1.0")
    def setContamination(self, value):
        """
        Sets the value of :py:attr:`contamination`.
        """
        return self._set(contamination=value)

    @since("2.1.0")
    def getContamination(self):
        """
        Gets the value of `contamination`
        """
        return self.getOrDefault(self.contamination)

    @since("2.1.0")
    def setBootstrap(self, value):
        """
        Sets the value of :py:attr:`bootstrap`.
        """
        return self._set(bootstrap=value)

    @since("2.1.0")
    def getBootstrap(self):
        """
        Gets the value of `bootstrap`
        """
        return self.getOrDefault(self.bootstrap)

    @since("2.1.0")
    def setApproxQuantileRelativeError(self, value):
        """
        Sets the value of :py:attr:`approxQuantileRelativeError`.
        """
        return self._set(approxQuantileRelativeError=value)

    @since("2.1.0")
    def getApproxQuantileRelativeError(self):
        """
        Gets the value of `approxQuantileRelativeError`
        """
        return self.getOrDefault(self.approxQuantileRelativeError)
