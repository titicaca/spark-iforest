from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
import os
import tempfile


if __name__ == "__main__":

    if "SPARK_HOME" in os.environ.keys():
        print("SPARK_HOME: ", os.environ['SPARK_HOME'])
    else:
        raise ValueError("Environment variable SPARK_HOME needs to be specified,"
                         " and make sure spark-iforest.jar is added into your lib path ($SPARK_HOME/jars")

    spark = SparkSession \
        .builder.master("local[*]") \
        .appName("IForestExample") \
        .getOrCreate()

    data = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([7.0, 9.0]),),
            (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)]

    df = spark.createDataFrame(data, ["features"])

    from pyspark_iforest.ml.iforest import *

    iforest = IForest(contamination=0.3, maxDepth=2)
    model = iforest.fit(df)

    model.hasSummary

    summary = model.summary

    summary.numAnomalies

    transformed = model.transform(df)

    rows = transformed.collect()

    temp_path = tempfile.mkdtemp()

    iforest_path = temp_path + "/iforest"

    iforest.save(iforest_path)

    loaded_iforest = IForest.load(iforest_path)

    model_path = temp_path + "/iforest_model"

    model.save(model_path)

    loaded_model = IForestModel.load(model_path)

    loaded_model.hasSummary

    loaded_model.transform(df).show()
