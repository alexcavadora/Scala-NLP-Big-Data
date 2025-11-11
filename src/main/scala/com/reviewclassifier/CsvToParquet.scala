package com.reviewclassifier

import org.apache.spark.sql.SparkSession

object CsvToParquet {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("CsvToParquet")
      .master("local[*]")
      .config("spark.driver.memory", "25g")
      .getOrCreate()

    println(s"Converting ${args(0)} to data/${args(1)}.parquet...")
    val testDf = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(s"${args(0)}")
    testDf.write.mode("overwrite").parquet(s"data/${args(1)}.parquet")
    println("Conversion of data complete.")

    spark.stop()
    println("Parquet conversion finished.")
  }
}


