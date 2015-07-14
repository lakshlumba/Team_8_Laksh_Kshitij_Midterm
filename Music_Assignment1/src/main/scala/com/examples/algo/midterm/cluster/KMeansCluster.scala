package com.examples.algo.midterm.cluster

import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.ChiSqSelector

/**
 * @author Laksh
 */
object KMeansCluster {
  
    def main(args: Array[String]) {
      
        // Load training data in LIBSVM format.
        val sc = new SparkContext(new SparkConf().setAppName("KMeansCluster"))
        
        // Load and parse the data
        val data = MLUtils.loadLibSVMFile(sc, "C:/Users/Laksh/Desktop/Bigdata/midterm/cluster/*")
        
        /*val discretizedData = data.map { lp =>
          LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map { x => x / 16 } ) )
        }
        // Create ChiSqSelector that will select 50 features
        val selector = new ChiSqSelector(50)
        // Create ChiSqSelector model (selecting features)
        val transformer = selector.fit(discretizedData)
        // Filter the top 50 features from each feature vector
        val filteredData = discretizedData.map { lp => 
          LabeledPoint(lp.label, transformer.transform(lp.features)) 
        }
        */
         val normalizer1 = new Normalizer()
        
        val parsedData = data.map { lp => normalizer1.transform(Vectors.dense(lp.features.toArray))}
        
        // Cluster the data into two classes using KMeans
        val numClusters = 2
        val numIterations = 20
        val clusters = KMeans.train(parsedData, numClusters, numIterations)
        
        // Evaluate clustering by computing Within Set Sum of Squared Errors
        val WSSSE = clusters.computeCost(parsedData)
        println("Within Set Sum of Squared Errors = " + WSSSE)
        
        // Save and load model
        clusters.save(sc, "myModelPath")
        val sameModel = KMeansModel.load(sc, "myModelPath")
    }
}