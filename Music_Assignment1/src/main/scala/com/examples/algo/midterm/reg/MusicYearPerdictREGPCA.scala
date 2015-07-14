package com.examples.algo.midterm.reg

import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
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
object MusicYearPerdictREGPCA {
  
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
          LabeledPoint(parts(0).toDouble, Vectors.dense(myList.map(_.toDouble))) 
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
        val numIterations = 100
        val stepSize = .00001
        val model = LinearRegressionWithSGD.train(training, numIterations, stepSize)
        val model_pca = LinearRegressionWithSGD.train(training_pca, numIterations, stepSize)
        
       
        
        // Evaluate model on training examples and compute training error
        val valuesAndPreds = parsedData.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }
        
        // Compute raw scores on the test set.
        val valuesAndPreds_pca = parsedData.map { point =>
          val prediction = model_pca.predict(point.features)
          (point.label, prediction)
        }
        
        // Get evaluation metrics.
        val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
        println("training Mean Squared Error = " + MSE)
        
        // Get evaluation metrics.
        val MSE_PCA = valuesAndPreds_pca.map{case(v, p) => math.pow((v - p), 2)}.mean()
        println("training Mean Squared Error with PCA= " + MSE_PCA)
        
        println("summary.mean = " + summary.mean) // a dense vector containing the mean value for each column
        println("summary.variance = " + summary.variance) // column-wise variance
        println("summary.numNonzeros = " + summary.numNonzeros) // number of nonzeros in each column
        println("summary.max = " + summary.max) // number of max in each column
        println("summary.min = " + summary.min) // number of max in each column
        
    }
}