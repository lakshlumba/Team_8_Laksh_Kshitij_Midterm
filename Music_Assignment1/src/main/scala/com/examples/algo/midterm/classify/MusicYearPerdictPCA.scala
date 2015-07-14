package com.examples.algo.midterm.classify

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.stat.Statistics._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.feature.ChiSqSelector
import org.apache.spark.mllib.feature.PCA

/**
 * @author Laksh
 */
object MusicYearPerdictPCA {
  
    def main(args: Array[String]) {
      
        // Load training data in LIBSVM format.
        val sc = new SparkContext(new SparkConf().setAppName("MusicYearPerdict"))
        
        val data = sc.textFile("C:/Users/Laksh/Desktop/Bigdata/midterm/YearPredictionMSD_C.csv")
               
        // Each sample in data1 will be normalized using $L^2$ norm.
        //val data1 = data.map(x => (x.label, normalizer1.transform(x.features))
        
        val normalizer1 = new Normalizer()

        val parsedData = data.map { line =>
          val parts = line.split(',')
          val myList = Array(parts(12), parts(78))
          LabeledPoint(parts(91).toDouble, normalizer1.transform(Vectors.dense(myList.map(_.toDouble)))) 
        }.cache()
        
        
        val vectorData = data.map { line =>
          val parts = line.split(',')
          val myList = Array(parts(12), parts(78))
          Vectors.dense(myList.map(_.toDouble)) 
        }
        
        val summary: MultivariateStatisticalSummary = Statistics.colStats(vectorData)
       
        
        // Split parsedData into training (90%) and test (10%).
        val splits = parsedData.randomSplit(Array(0.9, 0.1), seed = 11L)
        val training = splits(0).cache()
        val test = splits(1)
        
        // try PCA feature Reduction
        val pca = new PCA(training.first().features.size/2).fit(parsedData.map(_.features))
        val training_pca = training.map(p => p.copy(features = pca.transform(p.features)))
        val test_pca = test.map(p => p.copy(features = pca.transform(p.features)))
        
        
        
        
        // Run training algorithm to build the model
        val numIterations = 200
        val stepSize = .000001
        val regParam = .001
        val model = SVMWithSGD.train(training, numIterations, stepSize, regParam)
        val model_pca = SVMWithSGD.train(training_pca, numIterations, stepSize, regParam)
        
        // Clear the default threshold.
        model.clearThreshold()
        // Clear the default threshold.
        model_pca.clearThreshold()
        
        // Compute raw scores on the test set.
        val scoreAndLabels = test.map { point =>
          val score = model.predict(point.features)
          (score, point.label)
        }
        
        // Compute raw scores on the test set.
        val scoreAndLabels_pca = test_pca.map { point =>
          val score = model_pca.predict(point.features)
          (score, point.label)
        }
        
        // Get evaluation metrics.
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        val auROC = metrics.areaUnderROC()
        
        // Get evaluation metrics.
        val metrics_pca = new BinaryClassificationMetrics(scoreAndLabels_pca)
        val auROC_pca = metrics_pca.areaUnderROC()
        
        
        println("Area under ROC = " + auROC)
        println("Area under ROC with PCA = " + auROC_pca)
        
        println("summary.mean = " + summary.mean) // a dense vector containing the mean value for each column
        println("summary.variance = " + summary.variance) // column-wise variance
        println("summary.numNonzeros = " + summary.numNonzeros) // number of nonzeros in each column
        println("summary.max = " + summary.max) // number of max in each column
        println("summary.min = " + summary.min) // number of max in each column
        
    }
}