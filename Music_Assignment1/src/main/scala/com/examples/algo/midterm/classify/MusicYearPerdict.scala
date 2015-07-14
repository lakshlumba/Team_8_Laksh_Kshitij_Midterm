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
object MusicYearPerdict {
  
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
        }
        
        // Compute the top 10 principal components.
        //val pca = new PCA(10).fit(data.map(_.features))

        // Project vectors to the linear space spanned by the top 10 principal components, keeping the label
        //val projected = data.map(p => p.copy(features = pca.transform(p.features)))
        
        
        // Chi test cannot be performed , because there are more than 10000 differnt values in each column 
        // Discretize data in 16 equal bins since ChiSqSelector requires categorical features
        /*val discretizedData = data.map { line =>
          val parts = line.split(',')
          val myList = Array(parts(12), parts(78))
          LabeledPoint(parts(91).toDouble, Vectors.dense(myList.map(x => x.toDouble).toArray))
        }
        // Create ChiSqSelector that will select 50 features
        val selector = new ChiSqSelector(50)
        // Create ChiSqSelector model (selecting features)
        val transformer = selector.fit(discretizedData)
        // Filter the top 50 features from each feature vector
        val filteredData = discretizedData.map { line =>
          LabeledPoint(line.label, transformer.transform(line.features)) 
        }*/
        
        // The contingency table is constructed from the raw (feature, label) pairs and used to conduct
        // the independence test. Returns an array containing the ChiSquaredTestResult for every feature 
        // against the label.
        
        /*val obs = parsedData
        val featureTestResults = Statistics.chiSqTest(obs)
        var i = 1
        featureTestResults.foreach { result =>
            println(s"Column $i:\n$result")
            i += 1
        }*/
        
        
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
        
        // Run training algorithm to build the model
        val numIterations = 100
        val stepSize = .0000001
        val regParam = .001
        val model = SVMWithSGD.train(training, numIterations, stepSize, regParam)
        
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
        println("summary.mean = " + summary.mean) // a dense vector containing the mean value for each column
        println("summary.variance = " + summary.variance) // column-wise variance
        println("summary.numNonzeros = " + summary.numNonzeros) // number of nonzeros in each column
        println("summary.max = " + summary.max) // number of max in each column
        println("summary.min = " + summary.min) // number of max in each column
        
    }
}