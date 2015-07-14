package com.examples.algo.midterm.cluster

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.ChiSqSelector
import org.apache.spark.mllib.clustering.LDA

/**
 * @author Kshitij
 *
 */
// Latent Dirichlet Allocation
object LDA {

  def main(args: Array[String]) {

    // Load training data in LIBSVM format.
    val sc = new SparkContext(new SparkConf().setAppName("LDA"))

    // Load and parse the data
    val data = MLUtils.loadLibSVMFile(sc, "C:/Users/Laksh/Desktop/Bigdata/midterm/cluster/*")

    val parsedData = data.map { lp => Vectors.dense(lp.features.toArray) }

    // Index documents with unique IDs
    val corpus = parsedData.zipWithIndex.map(_.swap).cache()

    // Cluster the documents into three topics using LDA
    val ldaModel = new LDA().setK(3).run(corpus)

    // Output topics. Each is a distribution over words (matching word count vectors)
    println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")
    val topics = ldaModel.topicsMatrix
    for (topic <- Range(0, 3)) {
      print("Topic " + topic + ":")
      for (word <- Range(0, ldaModel.vocabSize)) { print(" " + topics(word, topic)); }
      println()
    }
  }
}