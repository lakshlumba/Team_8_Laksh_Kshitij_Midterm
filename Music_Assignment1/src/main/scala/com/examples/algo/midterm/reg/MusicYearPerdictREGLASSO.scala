package com.examples.algo.midterm.reg

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LassoWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

/**
 * @author Laksh
 */
object MusicYearPerdictREGLASSO {
  
    def main(args: Array[String]) {
      
        // Load training data in LIBSVM format.
        val sc = new SparkContext(new SparkConf().setAppName("MusicYearPerdict_Reg_Lasso"))
        
        val data = sc.textFile("C:/Users/Laksh/Desktop/Bigdata/midterm/YearPredictionMSD.txt")

        val parsedData = data.map { line =>
          val parts = line.split(',')
          val myList = Array(parts(12), parts(78))
          LabeledPoint(parts(0).toDouble, Vectors.dense(myList.map(_.toDouble)))
        }.cache()
        
        // Building the model
        val numIterations = 1000
        val stepSize = .00001
        val regParam = .01
        val model = LassoWithSGD.train(parsedData, numIterations,stepSize,regParam)
        
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
        sc.stop()
    }
}