package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Matrices, Vector}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalatest.flatspec._
import org.scalatest.matchers._

import scala.util.Random

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

    val delta: Double = 0.001

    lazy val yVector: DenseVector[Double] = LinearRegressionTest._yVector
    lazy val xyDataset: Dataset[_] = LinearRegressionTest._xyDataset
    lazy val expectedWeights: DenseVector[Double] = LinearRegressionTest._expectedWeights
    lazy val predictUDF: UserDefinedFunction = LinearRegressionTest._predictUDF

    private def validateTransformation(transformed: DataFrame): Unit = {

        transformed.columns should be(Seq("features", "label", "prediction"))
        transformed.collect().length should be(xyDataset.collect().length)

        val predicted = transformed.select("prediction").collect()

        (0 until 10).foreach(i => {
            predicted.toVector(i).getDouble(0) should be(yVector(i) +- delta)
        })
    }

    "Linear Regression Model" should "create prediction given weights" in {
        val model = new SimpleLinearRegressionModel(expectedWeights)
        val transformed = model.transform(xyDataset)

        validateTransformation(transformed)
    }

    "Linear Regression Model" should "work after write and load" in {
        //noinspection UnstableApiUsage
        val temporaryFolderPath = Files.createTempDir().getAbsolutePath

        new Pipeline()
            .setStages(Array(new LinearRegression()))
            .fit(xyDataset)
            .write
            .overwrite()
            .save(temporaryFolderPath)

        val transformed = PipelineModel.load(temporaryFolderPath).transform(xyDataset)

        validateTransformation(transformed)
    }

    "Linear Regression Model" should "make predictions correctly" in {
        import sqlContext.implicits._

        val randomData = Matrices
            .rand(30000, 3, Random.self)
            .rowIter
            .toSeq
            .map(x => Tuple1(x))
            .toDF("features")

        val dataset = randomData.withColumn("label", predictUDF(col("features")))
        val model = new LinearRegression().setMaxIter(8192).fit(dataset)

        (0 until model.weights.length).foreach(i => {
            model.weights(i) should be(expectedWeights(i) +- delta)
        })
    }
}

object LinearRegressionTest extends WithSpark {

    import sqlContext.implicits._

    private lazy val _expectedWeights: DenseVector[Double] = DenseVector(2.28, 0.42, -3.22)

    private lazy val _xMatrix: DenseMatrix[Double] = DenseMatrix.rand[Double](1000, 3)
    private lazy val _yVector: DenseVector[Double] = _xMatrix * _expectedWeights

    val _predictUDF: UserDefinedFunction = udf { features: Any =>
        val arr = features.asInstanceOf[Vector].toArray

        2.28 * arr.apply(0) + 0.42 * arr.apply(1) - 3.22 * arr.apply(2)
    }

    private lazy val _xyDataset: Dataset[_] = {
        val _xyMatrix = DenseMatrix.horzcat(_xMatrix, _yVector.asDenseMatrix.t)

        val vectorAssembler = new VectorAssembler()
            .setInputCols(Array("x1", "x2", "x3"))
            .setOutputCol("features")

        val transformed = vectorAssembler.transform(
            _xyMatrix(*, ::).iterator
                .map(row => Tuple4(row(0), row(1), row(2), row(3)))
                .toSeq
                .toDF("x1", "x2", "x3", "label")
        )

        transformed.select("features", "label")
    }
}
