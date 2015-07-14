package com.examples.algo

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

/**
 * @author Laksh
 */
object LinearSupportVectorL1 {
    def main(args: Array[String]) {
      
      // Load training data in LIBSVM format.
        val sc = new SparkContext(new SparkConf().setAppName("LinearSupportVectorL1"))
        
        val data = sc.textFile("C:/spark-1.4.0-bin-hadoop2.6/data/mllib/winequality-white.csv")
        
        val parsedData = data.map { line =>
          val parts = line.split(',')
          val myList = Array(parts(0), parts(1), parts(2), parts(3), parts(4), parts(5), parts(6), parts(7), parts(8), parts(9), parts(10))
          LabeledPoint(parts(11).toDouble, Vectors.dense(myList.map(_.toDouble)))
        }
        
        // Split parsedData into training (60%) and test (40%).
        val splits = parsedData.randomSplit(Array(0.8, 0.2), seed = 11L)
        val training = splits(0).cache()
        val test = splits(1)
        
        // Run training algorithm to build the model
        val numIterations = 1000
        val stepSize = .001
        val regParam = .01
        
        //val model = SVMWithSGD.train(training, numIterations, stepSize, regParam)
        
        val svmAlg = new SVMWithSGD()
        svmAlg.optimizer.setNumIterations(200).setRegParam(0.1).setUpdater(new L1Updater)
        val model = svmAlg.run(training)
        
        
        // Clear the default threshold.
        model.clearThreshold()
        
        // Compute raw scores on the test set.
        val scoreAndLabels = test.map { point =>
          val score = model.predict(point.features)
          (score, point.label)
        }
        
        // Get evaluation metrics.
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        val auROC = metrics.areaUnderROC()
        
        println("Area under ROC = " + auROC)
        
        // Save and load model
        model.save(sc, "myModelPath")
        val sameModel = SVMModel.load(sc, "myModelPath")
      
    }
}