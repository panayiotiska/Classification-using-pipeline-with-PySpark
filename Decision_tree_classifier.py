from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from sklearn.metrics import classification_report, confusion_matrix
from string import Template
from sys import argv
import pandas as pd

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("higgs_decision_tree_classifier")\
        .getOrCreate()

    # Load the data stored in csv format as a DataFrame.
    # Link to the original HIGGS dataset : https://archive.ics.uci.edu/ml/machine-learning-databases/00280/
    df = spark.read.format('com.databricks.spark.csv').options(header='false', inferSchema='true').load('Sample_HIGGS.csv')
    df = df.repartition(10)

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol='_c0', outputCol="label").fit(df)

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    features = df.columns[1:28]
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    df = assembler.transform(df)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",maxCategories=4).fit(df)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Create ParamGrid for Cross Validation
    dtparamGrid = (ParamGridBuilder() \
                .addGrid(dt.maxDepth, [5, 10, 15, 20]) \
                .addGrid(dt.maxBins, [3, 5, 10, 20]) \
                .addGrid(dt.impurity, ['entropy', 'gini']) \
                .addGrid(dt.minInstancesPerNode , [1, 2]) \
                .build())

    # Evaluate model
    dtevaluator = BinaryClassificationEvaluator(labelCol="label")

    # Create 5-fold CrossValidator
    dtcv = CrossValidator(estimator = pipeline,\
                        estimatorParamMaps = dtparamGrid, \
                        evaluator = dtevaluator, \
                        numFolds = 5)

    # Run cross validations
    dtcvModel = dtcv.fit(trainingData)
    print(dtcvModel)

    # Use test set here so we can measure the accuracy of our model on new data
    dtpredictions = dtcvModel.transform(testData)

    # cvModel uses the best model found from the Cross Validation
    # Evaluate best model
    # Print accuracy and AUC scores
    print('\nAccuracy:', dtevaluator.evaluate(dtpredictions))
    print('AUC:', BinaryClassificationMetrics(dtpredictions['label','prediction'].rdd).areaUnderROC)
    #ComputeModelStatistics().transform(dtpredictions)
    
    # Select columns for sklear's reports
    y_true = dtpredictions.select(['label']).collect()
    y_pred = dtpredictions.select(['prediction']).collect()

    # Reports
    my_tags = ['zero','one']

    # Assigne best results to strings
    line1 = "The best value for MaxDepth was: %s" % dtcvModel.bestModel.stages[-1]._java_obj.getMaxDepth()
    line2 = "The best value for MaxBins was: %s " % dtcvModel.bestModel.stages[-1]._java_obj.getMaxBins()
    line3 = "The best impurity method was: %s" % dtcvModel.bestModel.stages[-1]._java_obj.getImpurity()
    line4 = "The best value for MinInstancesPerNode was: %s" % dtcvModel.bestModel.stages[-1]._java_obj.getMinInstancesPerNode()
    best_results = (line1,"\n",line2,"\n",line3,"\n",line4,"\n")

    # Print reports
    print('---------------------------------------------------------------------------')
    print(classification_report(y_true, y_pred,target_names=my_tags, output_dict=True))
    print('---------------------------------------------------------------------------')
    print(confusion_matrix(y_true, y_pred))
    print('---------------------------------------------------------------------------')
    print(best_results)

    #Write reports to files
    f = open("best_results.txt", "w")
    f.write(best_results)
    f.close()

    report = classification_report(y_true, y_pred,target_names=my_tags, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(r'report.csv', index = True, float_format="%.3f")

    confusion = confusion_matrix(y_true, y_pred)
    df_confusion = pd.DataFrame(confusion)
    df_confusion.to_csv('cm.csv')

    spark.stop()
