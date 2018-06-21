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

def train_cv_model(modelDataframe):
    cv = CountVectorizer(inputCol="udf_results", outputCol="features", binary=True, minDF=5.0)
    model = cv.fit(modelDataframe)
    model.write().overwrite().save("models/cvModel")

def transform_model(sqlContext, modelDataframe, labels):
    labels.registerTempTable("labelsTable")
    # Load the CV model
    model = CountVectorizerModel.load("models/cvModel")
    # Transform the data frame
    transformedDf = model.transform(modelDataframe)
    transformedDf.registerTempTable("transformedDfTable")
    transformedDf = sqlContext.sql("SELECT transformedDfTable.comments_id AS id, transformedDfTable.comments_body AS body, transformedDfTable.comments_author AS author, transformedDfTable.comments_created_utc AS created_utc, transformedDfTable.comments_subreddit_id AS subreddit_id, transformedDfTable.comments_link_id AS link_id, transformedDfTable.comments_parent_id AS parent_id, transformedDfTable.comments_score AS score, transformedDfTable.comments_controversiality AS controversiality, transformedDfTable.comments_gilded AS gilded, transformedDfTable.udf_results AS udf_results, transformedDfTable.features AS features, IF(labelsTable.label=1, 1, 0) AS pos_label, IF(labelsTable.label=-1, 1, 0) AS neg_label FROM transformedDfTable INNER JOIN labelsTable ON transformedDfTable.comments_id = labelsTable.label_id")

    return transformedDf

def create_models(sqlContext, modelDataframe):
    modelDataframe.registerTempTable("modelDataframeTable")

    # Create dataframes to use on the positive and negative models
    pos = sqlContext.sql("SELECT pos_label AS label, features FROM modelDataframeTable")
    neg = sqlContext.sql("SELECT neg_label AS label, features FROM modelDataframeTable")

    # Initialize two logistic regression models.
    # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    poslr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10).setThreshold(0.2)
    neglr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10).setThreshold(0.25)
    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator()
    negEvaluator = BinaryClassificationEvaluator()
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,
        numFolds=2)
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=2)
    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50
    posTrain, posTest = pos.randomSplit([0.5, 0.5])
    negTrain, negTest = neg.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)

    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.write().overwrite().save("models/posModel")
    negModel.write().overwrite().save("models/negModel")

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

    # Fit the CountVectorizer model
    if(not os.path.exists("models/cvModel")):
        train_cv_model(modelDataframe)

    # Use model to transform the data
    modelDataframe = transform_model(sqlContext, modelDataframe, labels)

    if(not os.path.exists("models/negModel") or not os.path.exists("models/posModel")):
        create_models(sqlContext, modelDataframe)

    # Load the positive and negative models back in
    posModel = CrossValidatorModel.load("models/posModel")
    negModel = CrossValidatorModel.load("models/negModel")



if __name__ == "__main__":
    conf = SparkConf().setAppName("SuperBowl Sentiment Analysis")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    import cleantext
    main(sqlContext)
