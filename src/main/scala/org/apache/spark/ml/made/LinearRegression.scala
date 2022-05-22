package org.apache.spark.ml.made

import breeze.linalg.functions.euclideanDistance
import breeze.linalg.{DenseVector, sum}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{VectorUDT, Vector => MlVector, Vectors => MlVectors}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.mllib.linalg.{Vectors => MllibVectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait LinearRegressionParams extends PredictorParams with HasMaxIter with HasTol {

    val learningRate: Param[Double] = new DoubleParam(this, "learningRate", "Learning rate")

    def getLearningRate: Double = $(learningRate)

    def setLearningRate(value: Double): this.type = set(learningRate, value)

    setDefault(learningRate -> 0.05)

    def setMaxIter(value: Int): this.type = set(maxIter, value)

    setDefault(maxIter -> 100000)

    def setTol(value: Double): this.type = set(tol, value)

    setDefault(tol -> 0.0000001)

    def setPredictionCol(value: String): this.type = set(predictionCol, value)

    def setLabelCol(value: String): this.type = set(labelCol, value)

    def setFeaturesCol(value: String): this.type = set(featuresCol, value)

    protected def validateAndTransform(schema: StructType): StructType = {
        SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT())

        if (schema.fieldNames.contains($(predictionCol)) == false) {
            SchemaUtils.appendColumn(schema, schema($(featuresCol)).copy(name = $(predictionCol)))
        } else {
            SchemaUtils.checkColumnType(schema, $(predictionCol), DoubleType)

            return schema
        }
    }
}

class LinearRegression(override val uid: String)
    extends Estimator[SimpleLinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {

    def this() = this(Identifiable.randomUID("linearRegression"))

    override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)

    override def transformSchema(schema: StructType): StructType = validateAndTransform(schema)

    override def fit(dataset: Dataset[_]): SimpleLinearRegressionModel = {

        implicit val encoder: Encoder[MlVector] = ExpressionEncoder()

        val outputColumnName = "output column"

        val vectors = {
            val assembler = new VectorAssembler().setInputCols(Array($(featuresCol), $(labelCol))).setOutputCol(outputColumnName)

            assembler.transform(dataset).select(outputColumnName).as[MlVector]
        }

        val weightsCount = vectors.first().size - 1
        var previousWeights = DenseVector.fill(weightsCount, Double.PositiveInfinity)
        val currentWeights = DenseVector.fill(weightsCount, 0d)

        var currentIteration = 0

        while (currentIteration < $(maxIter) && euclideanDistance(previousWeights, currentWeights) > $(tol)) {

            currentIteration += 1

            val summary = vectors.rdd.mapPartitions((data: Iterator[MlVector]) => {
                val summarizer = new MultivariateOnlineSummarizer()

                data.foreach(row => {
                    val x = row.asBreeze(0 until weightsCount).toDenseVector
                    val weightsDelta = x * (x.dot(currentWeights) - row.asBreeze(-1))

                    summarizer.add(MllibVectors.fromBreeze(weightsDelta))
                })

                Iterator(summarizer)
            }).reduce(_ merge _)

            previousWeights = currentWeights.copy
            currentWeights -= $(learningRate) * summary.mean.asBreeze
        }

        copyValues(new SimpleLinearRegressionModel(currentWeights).setParent(this))
    }
}

class SimpleLinearRegressionModel(override val uid: String, val weights: DenseVector[Double])
    extends Model[SimpleLinearRegressionModel] with LinearRegressionParams with MLWritable {

    def this(weights: DenseVector[Double]) = this(Identifiable.randomUID("linearRegressionModel"), weights)

    override def copy(extra: ParamMap): SimpleLinearRegressionModel = copyValues(new SimpleLinearRegressionModel(weights), extra)

    override def transformSchema(schema: StructType): StructType = validateAndTransform(schema)

    override def transform(dataset: Dataset[_]): DataFrame = {
        val transformUdf = {
            dataset.sqlContext.udf.register(
                uid + "_transform",
                (x: MlVector) => {
                    sum(x.asBreeze.toDenseVector * weights(0 until weights.length))
                }
            )
        }

        dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
    }

    override def write: MLWriter = new DefaultParamsWriter(this) {

        override protected def saveImpl(path: String): Unit = {
            super.saveImpl(path)

            val vectors = Tuple1(MlVectors.fromBreeze(weights))

            sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
        }
    }
}

object SimpleLinearRegressionModel extends MLReadable[SimpleLinearRegressionModel] {

    override def read: MLReader[SimpleLinearRegressionModel] = new MLReader[SimpleLinearRegressionModel] {

        override def load(path: String): SimpleLinearRegressionModel = {
            val metadata = DefaultParamsReader.loadMetadata(path, sc)
            val vectors = sqlContext.read.parquet(path + "/vectors")

            implicit val encoder: Encoder[MlVector] = ExpressionEncoder()

            val weights = vectors.select(vectors("_1").as[MlVector]).first().asBreeze.toDenseVector
            val model = new SimpleLinearRegressionModel(weights)

            metadata.getAndSetParams(model)

            return model
        }
    }
}
