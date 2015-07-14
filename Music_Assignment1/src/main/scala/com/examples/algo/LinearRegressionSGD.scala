package com.examples.algo

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

/**
 * @author Laksh
 */
object LinearRegressionSGD {
  
    def main(args: Array[String]) {
      
        
        val sc = new SparkContext(new SparkConf().setAppName("LinearRegressionSGD"))
        val data = sc.textFile("C:/spark-1.4.0-bin-hadoop2.6/data/mllib/winequality-white-WB.csv")

        val parsedData = data.map { line =>
          val parts = line.split(',')
          val myList = Array(parts(0), parts(1), parts(2), parts(3), parts(4), parts(5), parts(6), parts(7), parts(8), parts(9), parts(10))
          LabeledPoint(parts(11).toDouble, Vectors.dense(myList.map(_.toDouble)))
        }.cache()
        
        // Building the model
        val numIterations = 1000
        val stepSize = .00001
        val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize)
        
        // Evaluate model on training examples and compute training error
        val valuesAndPreds = parsedData.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }
        val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
        println("training Mean Squared Error = " + MSE)
        
        // Save and load model
        model.save(sc, "myModelPath")
        val sameModel = LinearRegressionModel.load(sc, "myModelPath")
      
    }
}