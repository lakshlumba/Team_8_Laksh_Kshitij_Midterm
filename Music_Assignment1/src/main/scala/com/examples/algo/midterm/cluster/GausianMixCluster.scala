package com.examples.algo.midterm.cluster

import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.GaussianMixtureModel
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
object GausianMixCluster {
  
    def main(args: Array[String]) {
      
        // Load training data in LIBSVM format.
        val sc = new SparkContext(new SparkConf().setAppName("KMeansCluster"))
        
        // Load and parse the data
        val data = MLUtils.loadLibSVMFile(sc, "C:/Users/Laksh/Desktop/Bigdata/midterm/cluster/BBC.txt")
        
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
        val parsedData = data.map { lp => Vectors.dense(lp.features.toArray)}
        
        // Cluster the data into two classes using GaussianMixture
        val gmm = new GaussianMixture().setK(20).run(parsedData)
        
        // Save and load model
        gmm.save(sc, "myGMMModel")
        val sameModel = GaussianMixtureModel.load(sc, "myGMMModel")
        
        // output parameters of max-likelihood model
        for (i <- 0 until gmm.k) {
          println("weight=%f\nmu=%s\nsigma=\n%s\n" format
            (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
        }
    }
}