package com.examples.algo.midterm.df

import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrameReader
import org.apache.spark.sql.Column
import org.apache.spark.sql.DataFrameNaFunctions
import com.sun.xml.internal.bind.v2.schemagen.xmlschema.List
import org.apache.spark.sql.DataFrame
import com.google.common.collect.ImmutableMap
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * @author Kshitij
 */

object AdultDataSet {

  case class AdultData(age: Int, workclass: String, fnlwgt: Int, education: String, educationNum: Int, maritalStatus: String,
    occupation: String, relationship: String, race: String, sex: String,
    capitalGain: Int, capitalLoss: Int, hpw: Int, nativeCountry: String, salary: String)

  //  class CsvFile
  def main(args: Array[String]) {

              val sc = new SparkContext(new SparkConf().setAppName("AdultDataSet"))
              val sqlContext = new org.apache.spark.sql.SQLContext(sc)
              import sqlContext.implicits._

    
            //create an RDD of above object and register it as table
              val readData = sc.textFile("C:/Users/Laksh/Desktop/Bigdata/midterm/AdultDataSet.csv")
              
            val data = readData.map(_.split(",")).map(p => AdultData(p(0).trim.toInt, p(1), p(2).trim.toInt, p(3),
              p(4).trim().toInt, p(5), p(6), p(7), p(8), p(9), p(10).trim().toInt, p(11).trim().toInt,
              p(12).trim().toInt, p(13), p(14))).toDF()
              
              var cat = data.na.replace("workclass", Map(" ?" -> " Private"))
              var mat = cat.na.replace("occupation", Map(" ?" -> " Prof-speciality"))
              var processedData = mat.na.replace("nativeCountry", Map(" ?" -> " United-States"))
              
              val indexer1 = new StringIndexer().setInputCol("workclass").setOutputCol("workclassIndex").fit(processedData)
              val indexed1 = indexer1.transform(processedData)
              val encoder1 = new OneHotEncoder().setInputCol("workclassIndex").setOutputCol("workclassVec")
              val encoded1 = encoder1.transform(indexed1)
              
              val indexer2 = new StringIndexer().setInputCol("education").setOutputCol("educationIndex").fit(encoded1)
              val indexed2 = indexer2.transform(encoded1)
              val encoder2 = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("educationVec")
              val encoded2 = encoder2.transform(indexed2)
              
              val indexer3 = new StringIndexer().setInputCol("maritalStatus").setOutputCol("maritalStatusIndex").fit(encoded2)
              val indexed3 = indexer3.transform(encoded2)
              val encoder3 = new OneHotEncoder().setInputCol("maritalStatusIndex").setOutputCol("maritalStatusVec")
              val encoded3 = encoder3.transform(indexed3)
              
              val indexer4 = new StringIndexer().setInputCol("occupation").setOutputCol("occupationIndex").fit(encoded3)
              val indexed4 = indexer4.transform(encoded3)
              val encoder4 = new OneHotEncoder().setInputCol("occupationIndex").setOutputCol("occupationVec")
              val encoded4 = encoder4.transform(indexed4)
              
              val indexer5 = new StringIndexer().setInputCol("relationship").setOutputCol("relationshipIndex").fit(encoded4)
              val indexed5 = indexer5.transform(encoded4)
              val encoder5 = new OneHotEncoder().setInputCol("relationshipIndex").setOutputCol("relationshipVec")
              val encoded5 = encoder5.transform(indexed5)
              
              val indexer6 = new StringIndexer().setInputCol("race").setOutputCol("raceIndex").fit(encoded5)
              val indexed6 = indexer6.transform(encoded5)
              val encoder6 = new OneHotEncoder().setInputCol("raceIndex").setOutputCol("raceVec")
              val encoded6 = encoder6.transform(indexed6)
              
              val indexer7 = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex").fit(encoded6)
              val indexed7 = indexer7.transform(encoded6)
              val encoder7 = new OneHotEncoder().setInputCol("sexIndex").setOutputCol("sexVec")
              val encoded7 = encoder7.transform(indexed7)
              
              val indexer8 = new StringIndexer().setInputCol("nativeCountry").setOutputCol("nativeCountryIndex").fit(encoded7)
              val indexed8 = indexer8.transform(encoded7)
              val encoder8 = new OneHotEncoder().setInputCol("nativeCountryIndex").setOutputCol("nativeCountryVec")
              val encoded8 = encoder8.transform(indexed8)
              
              val indexer9 = new StringIndexer().setInputCol("salary").setOutputCol("salaryIndex").fit(encoded8)
              val indexed9 = indexer9.transform(encoded8)
              val encoder9 = new OneHotEncoder().setInputCol("salaryIndex").setOutputCol("salaryVec")
              val encoded9 = encoder9.transform(indexed9)
              
              val stringJson = encoded9.toJSON
              val stringJsonArray = stringJson.collect()
              
              val trainingData = stringJson.map { line =>
                LabeledPoint(stringJsonArray.last.toDouble , Vectors.dense(stringJsonArray.map(_.toDouble)))
              }
                  
              
              //create an RDD of above object and register it as table
              val readTestData = sc.textFile("C:/Users/Laksh/Desktop/Bigdata/midterm/Adult.Test.csv")
              
            val tdata = readTestData.map(_.split(",")).map(p => AdultData(p(0).trim.toInt, p(1), p(2).trim.toInt, p(3),
              p(4).trim().toInt, p(5), p(6), p(7), p(8), p(9), p(10).trim().toInt, p(11).trim().toInt,
              p(12).trim().toInt, p(13), p(14))).toDF()
              
              var tcat = tdata.na.replace("workclass", Map(" ?" -> " Private"))
              var tmat = tcat.na.replace("occupation", Map(" ?" -> " Prof-speciality"))
              var tprocessedData = tmat.na.replace("nativeCountry", Map(" ?" -> " United-States"))
              
              val tindexer1 = new StringIndexer().setInputCol("workclass").setOutputCol("workclassIndex").fit(tprocessedData)
              val tindexed1 = tindexer1.transform(tprocessedData)
              val tencoder1 = new OneHotEncoder().setInputCol("workclassIndex").setOutputCol("workclassVec")
              val tencoded1 = tencoder1.transform(tindexed1)
              
              val tindexer2 = new StringIndexer().setInputCol("education").setOutputCol("educationIndex").fit(tencoded1)
              val tindexed2 = tindexer2.transform(tencoded1)
              val tencoder2 = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("educationVec")
              val tencoded2 = tencoder2.transform(tindexed2)
              
              val tindexer3 = new StringIndexer().setInputCol("maritalStatus").setOutputCol("maritalStatusIndex").fit(tencoded2)
              val tindexed3 = tindexer3.transform(tencoded2)
              val tencoder3 = new OneHotEncoder().setInputCol("maritalStatusIndex").setOutputCol("maritalStatusVec")
              val tencoded3 = tencoder3.transform(tindexed3)
              
              val tindexer4 = new StringIndexer().setInputCol("occupation").setOutputCol("occupationIndex").fit(tencoded3)
              val tindexed4 = tindexer4.transform(tencoded3)
              val tencoder4 = new OneHotEncoder().setInputCol("occupationIndex").setOutputCol("occupationVec")
              val tencoded4 = tencoder4.transform(tindexed4)
              
              val tindexer5 = new StringIndexer().setInputCol("relationship").setOutputCol("relationshipIndex").fit(tencoded4)
              val tindexed5 = tindexer5.transform(tencoded4)
              val tencoder5 = new OneHotEncoder().setInputCol("relationshipIndex").setOutputCol("relationshipVec")
              val tencoded5 = tencoder5.transform(tindexed5)
              
              val tindexer6 = new StringIndexer().setInputCol("race").setOutputCol("raceIndex").fit(tencoded5)
              val tindexed6 = tindexer6.transform(tencoded5)
              val tencoder6 = new OneHotEncoder().setInputCol("raceIndex").setOutputCol("raceVec")
              val tencoded6 = tencoder6.transform(tindexed6)
              
              val tindexer7 = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex").fit(tencoded6)
              val tindexed7 = tindexer7.transform(tencoded6)
              val tencoder7 = new OneHotEncoder().setInputCol("sexIndex").setOutputCol("sexVec")
              val tencoded7 = tencoder7.transform(tindexed7)
              
              val tindexer8 = new StringIndexer().setInputCol("nativeCountry").setOutputCol("nativeCountryIndex").fit(tencoded7)
              val tindexed8 = tindexer8.transform(tencoded7)
              val tencoder8 = new OneHotEncoder().setInputCol("nativeCountryIndex").setOutputCol("nativeCountryVec")
              val tencoded8 = tencoder8.transform(tindexed8)
              
              val tindexer9 = new StringIndexer().setInputCol("salary").setOutputCol("salaryIndex").fit(tencoded8)
              val tindexed9 = tindexer9.transform(tencoded8)
              val tencoder9 = new OneHotEncoder().setInputCol("salaryIndex").setOutputCol("salaryVec")
              val tencoded9 = tencoder9.transform(tindexed9)
              
              
              val stringTJson = tencoded9.toJSON
              val stringArrayTest = stringTJson.collect()
              
              val testData = readTestData.map { line =>
                LabeledPoint(stringArrayTest.last.toDouble , Vectors.dense(stringArrayTest.map(x => x.toDouble).toArray))
              }
              
              // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
              // Create a LogisticRegression instance.  This instance is an Estimator.
              val lr = new LogisticRegression()
              // Print out the parameters, documentation, and any default values.
              println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

              // We may set parameters using setter methods.
              lr.setMaxIter(10)
                .setRegParam(0.01)
              
              // Learn a LogisticRegression model.  This uses the parameters stored in lr.
              val model1 = lr.fit(trainingData.df)
              // Since model1 is a Model (i.e., a Transformer produced by an Estimator),
              // we can view the parameters it used during fit().
              // This prints the parameter (name: value) pairs, where names are unique IDs for this
              // LogisticRegression instance.
              println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)
              
              // We may alternatively specify parameters using a ParamMap,
              // which supports several methods for specifying parameters.
              val paramMap = ParamMap(lr.maxIter -> 20)
              paramMap.put(lr.maxIter, 30) // Specify 1 Param.  This overwrites the original maxIter.
              paramMap.put(lr.regParam -> 0.1, lr.threshold -> 0.55) // Specify multiple Params.
              
              // One can also combine ParamMaps.
              val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability") // Change output column name
              val paramMapCombined = paramMap ++ paramMap2
              
              // Now learn a new model using the paramMapCombined parameters.
              // paramMapCombined overrides all parameters set earlier via lr.set* methods.
              val model2 = lr.fit(trainingData.df, paramMapCombined)
              println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)
              
              // Prepare test data.
              // Make predictions on test data using the Transformer.transform() method.
              // LogisticRegression.transform will only use the 'features' column.
              // Note that model2.transform() outputs a 'myProbability' column instead of the usual
              // 'probability' column since we renamed the lr.probabilityCol parameter previously.
              model2.transform(testData.df)
                .select("features", "label", "myProbability", "prediction")
                .collect()
                .foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
                  println(s"($features, $label) -> prob=$prob, prediction=$prediction")
             }
             
                         
  }
}