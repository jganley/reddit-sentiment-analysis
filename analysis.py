from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os

def write_parquet(sqlContext):
    # Read from JSON
    comments = sqlContext.read.json("superbowl_comments.json")

    # Write the Parquets
    comments.write.parquet("superbowl_comments.parquet")

def create_dataframe(sqlContext, comments, labels):

    comments.registerTempTable("commentsTable")
    labels.registerTempTable("labelsTable")

    labeledData = sqlContext.sql("SELECT commentsTable.* FROM commentsTable INNER JOIN labelsTable ON commentsTable.comments_id = labelsTable.label_id")

    def parse_text(text):
        return cleantext.sanitize(text)

    parse_udf = udf(parse_text, ArrayType(StringType()))
    labeledData = labeledData.withColumn("udf_results", parse_udf(col("comments_body")))

    return labeledData

def main(sqlContext):

    # Check if the parquet file has already been written
    if not os.path.exists("superbowl_comments.parquet"):
        write_parquet(sqlContext)

    # Read the parquets
    comments = sqlContext.read.parquet("superbowl_comments.parquet")
    comments.registerTempTable("commentsTable")

    # Read the labels csv
    labels = sqlContext.read.format('csv').options(header='true', inferSchema='true').load("labels.csv")
    labels.registerTempTable("labelsTable")

    # Create Dataframe to Train the Model
    modelDataframe = create_dataframe(sqlContext, comments, labels)

    modelDataframe.show()

if __name__ == "__main__":
    conf = SparkConf().setAppName("SuperBowl Sentiment Analysis")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    import cleantext
    main(sqlContext)
