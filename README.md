# Classification-using-pipeline-with-PySpark
Machine learning task in big data, using a distributed system for training a big dataset with the use of PySpark library.

To be able to use Spark, first upload the HIGGS.CSV file to the Hadoop File System (HDFS), add the names of the slaves and the master in the slaves configuration file of Spark and pass as a parameter in the program the location of the csv in HDFS. Both Hadoop and HDFS have to be active.

-	Command for one cluster: ~/spark/bin/spark-submit --master spark://master:7077 ~/…/Decision_tree_classifier.py
-	Command for several clusters: ~/spark/bin/spark-submit --master spark://master:7077 ~/…/ Decision_tree_classifier.py
hdfs://master:9000/user/user/input/Sample_HIGGS.csv

### Results for the original dataset

- The best value for MaxDepth was: 65
- The best value for MaxBins was: 603483
- The best impurity method was: gini
- The best value for MinInstancesPerNode was: 1

#### Confusion Matrix

![alt text](https://raw.githubusercontent.com/panayiotiska/Classification-using-pipeline-with-PySpark/main/confusion_matrix.png)

#### Run Times

![alt text](https://raw.githubusercontent.com/panayiotiska/Classification-using-pipeline-with-PySpark/main/run_times.png)
