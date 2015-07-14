package com.examples.algo

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import scala.reflect.runtime.universe

/**
 * @author Laksh
 */
object Test {  
  
      case class Person(name: String, age: Int)
      
      def main(args: Array[String]) {
  
        val sc = new SparkContext(new SparkConf().setAppName("Test"))    
        // sc is an existing SparkContext.
        val sqlContext = new org.apache.spark.sql.SQLContext(sc)
        // this is used to implicitly convert an RDD to a DataFrame.
        import sqlContext.implicits._
        
        // Define the schema using a case class.
        // Note: Case classes in Scala 2.10 can support only up to 22 fields. To work around this limit,
        // you can use custom classes that implement the Product interface.
        
        
        // Create an RDD of Person objects and register it as a table.
        val people = sc.textFile("C:/spark-1.4.0-bin-hadoop2.6/examples/src/main/resources/people.txt").map(_.split(",")).map(p => Person(p(0), p(1).trim.toInt)).toDF()
        people.registerTempTable("people")
        
        // SQL statements can be run by using the sql methods provided by sqlContext.
        val teenagers = sqlContext.sql("SELECT name, age FROM people WHERE age >= 13 AND age <= 19")
        
        // The results of SQL queries are DataFrames and support all the normal RDD operations.
        // The columns of a row in the result can be accessed by field index:
        teenagers.map(t => "Name: " + t(0)).collect().foreach(println)
        
        // or by field name:
        teenagers.map(t => "Name: " + t.getAs[String]("name")).collect().foreach(println)
        
        // row.getValuesMap[T] retrieves multiple columns at once into a Map[String, T]
        teenagers.map(_.getValuesMap[Any](List("name", "age"))).collect().foreach(println)
        // Map("name" -> "Justin", "age" -> 19)
      }

}