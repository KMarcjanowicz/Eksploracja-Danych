package isi.ed;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.feature.VectorAssembler;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static isi.ed.Plotting.plotObjectiveHistory;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lit;

public class LinearRegressionClass {

    private SparkSession spark = null;

    LinearRegressionClass(){
        spark = SparkSession.builder()
                .appName("Functions")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());
    }

    public Dataset<Row> Load(int number){

        String file = "xy-";
        if(number < 10){
            file += "00" + number + ".csv";
        }
        else if(number < 100){
            file += "0" + number + ".csv";
        }

        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "X",
                        DataTypes.DoubleType,
                        false),
                DataTypes.createStructField(
                        "Y",
                        DataTypes.DoubleType,
                        false)
        });

        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("src/main/resources/functions/" + file);

        System.out.println("Excerpt of the " + file + " content:");

        df.show(20);
        System.out.println(file + " schema:");
        df.printSchema();

        return df;
    }

    public Dataset<Row> Process(Dataset<Row> _df){

        VectorAssembler va = new VectorAssembler();
        String[] names = {"X"};
        va.setInputCols(names);
        va.setOutputCol("features");
        var df = va.transform(_df);

        System.out.println("Processed to:");
        df.show(5);
        System.out.println("Processed dataframe's schema:");
        df.printSchema();

        return df;
    }

    public LinearRegressionModel LinearFit(Dataset<Row> _df_trans, double RegParam, double ElasticParam){
        LinearRegression lr = new LinearRegression()
                .setMaxIter(10)
                .setRegParam(RegParam)
                .setElasticNetParam(ElasticParam)
                .setFeaturesCol("features")
                .setLabelCol("Y");

        // Fit the model.
        LinearRegressionModel lrModel = lr.fit(_df_trans);
        System.out.println("Coefficients: ");
        System.out.println(lrModel.coefficients());
        System.out.println("Intercept: ");
        System.out.println(lrModel.intercept());

        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show(100);
        System.out.println("MSE: " + trainingSummary.meanSquaredError());
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("MAE: " + trainingSummary.meanAbsoluteError());
        System.out.println("r2: " + trainingSummary.r2());

        //System.out.println(Arrays.toString(trainingSummary.objectiveHistory()));
        Double[] doubleArray = ArrayUtils.toObject(trainingSummary.objectiveHistory());
        List<Double> list = Arrays.asList(doubleArray);
        plotObjectiveHistory(list);

        return lrModel;
    }

    public Dataset<Row> OLS(Dataset<Row> df){
        // Add a new column to the dataset
        Dataset<Row> datasetWithNewColumn = df.withColumn("ones", lit("1")).selectExpr("ones", "*");;
        datasetWithNewColumn.show(20);
        return datasetWithNewColumn;
    }
}
