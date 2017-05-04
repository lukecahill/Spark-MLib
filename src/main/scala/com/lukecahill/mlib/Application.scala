package com.lukecahill.mlib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import java.lang.Math

/**
  * Created by Administrator on 04/05/2017.
  */
object Application {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val sc = new SparkContext("local[*]", "Machine Learning")

    val users = sc.textFile("file:///Users/Luke/Downloads/ml-1m/users.dat").map(x => x.split("::"))
    val movies = sc.textFile("file:///Users/Luke/Downloads/ml-1m/movies.dat").map(x => x.split("::"))
    val ratings = sc.textFile("file:///Users/Luke/Downloads/ml-1m/ratings.dat").map(x => x.split("::"))

    case class User(userId: Int, gender: String, age: Int, occupation: Int, zip: String)
    case class Movie(movieId: Int, title: String)

    val userRdd = users.map(x => User(x(0).toInt, x(1).toString, x(2).toInt, x(3).toInt, x(4).toString))
    val movieRdd = movies.map(x => Movie(x(0).toInt, x(1).toString))
    val ratingsRdd = ratings.map(x => {
      var roundedRating = Math.round(x(2).toDouble)
      if(roundedRating == 0) roundedRating = 1
      Rating(x(0).toInt, x(1).toInt, roundedRating)
    })

    val splits = ratingsRdd.randomSplit(Array(0.8, 0.2), 0L)
    val trainingRatings = splits(0).cache()
    val testRatings = splits(1).cache()

    val model = (new ALS().setRank(10).setIterations(10).run(trainingRatings))
    val blankMovieUserId = testRatings.map {
      case Rating(user, movie, ratings) => (user, movie)
    }

    val predictionsForTest = model.predict(blankMovieUserId)
    println(predictionsForTest.first())

    val testKeys = testRatings.map {
      case Rating(user, movie, rating) => ((user, movie), rating)
    }

    val predictionKeys = predictionsForTest.map {
      case Rating(user, movie, rating) => ((user, movie), rating)
    }

    val joinedRdd = testKeys.join(predictionKeys)

    val MAE = joinedRdd.map {
      case ((user, product), (testRating, predRating)) =>
        val err = (testRating - predRating)
        Math.abs(err)
    }.mean()

    println("The MAE is: " + MAE)

    val falsePositives = (joinedRdd.filter {
      case ((user, product), (testRating, predRating)) =>
        (testRating <= 1 && predRating >= 4)
    })

    val percentage = ((falsePositives.count().toFloat / joinedRdd.count().toFloat) * 100).toFloat
    println("The count of false positive is: " + falsePositives.count() + " and the percentage is: " + percentage.toFloat)

    val topRecs = model.recommendProducts(222, 10)
    val movieTitles = movies.map(x => (x(0), x(1))).collectAsMap()
    println("The top recommendations are: ")
    topRecs.map(x => (movieTitles(x.product.toString), x.rating)).foreach(println)
  }
}
