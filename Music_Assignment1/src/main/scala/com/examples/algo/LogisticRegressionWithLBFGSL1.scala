package com.examples.algo

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

/**
 * @author Laksh
 */
object LogisticRegressionLBFGSL1 {
  
    def main(args: Array[String]) {
      
        // Load training data in LIBSVM format.
        val sc = new SparkContext(new SparkConf().setAppName("LogisticRegressionWithLBFGSL1"))
        
        val data = sc.textFile("C:/spark-1.4.0-bin-hadoop2.6/data/mllib/winequality-white.csv")
        
        val parsedData = data.map { line =>
          val parts = line.split(',')
          val myList = Array(parts(0), parts(1), parts(2), parts(3), parts(4), parts(5), parts(6), parts(7), parts(8), parts(9), parts(10))
          LabeledPoint(parts(11).toDouble, Vectors.dense(myList.map(_.toDouble)))
        }
        
        // Split data into training (60%) and test (40%).
        val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
        val training = splits(0).cache()
        val test = splits(1)
        
        // Run training algorithm to build the model       
        val model = new LogisticRegressionWithLBFGS().setNumClasses(10).run(training)
        
        // Compute raw scores on the test set.
        val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
          val prediction = model.predict(features)
          (prediction, label)
        }
        
        // Get evaluation metrics.
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val precision = metrics.precision
        println("Precision = " + precision)
        
        // Save and load model
        model.save(sc, "myModelPath")
        val sameModel = LogisticRegressionModel.load(sc, "myModelPath")
      
    }
}