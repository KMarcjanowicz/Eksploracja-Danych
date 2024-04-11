package isi.ed;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

import static isi.ed.Plotting.plotObjectiveHistory;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.try_multiply;

public class LinearRegressionPolynomialFeatures {

    private SparkSession spark = null;

    LinearRegressionPolynomialFeatures(){
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
                .load("src/main/resources/" + file);

        System.out.println("Excerpt of the " + file + " content:");

        df.show(5);
        System.out.println(file + " schema:");
        df.printSchema();

        return df;
    }

    public Dataset<Row> AddXSquared(Dataset<Row> _df){

        Dataset<Row> df = null;

        df = _df.withColumn("X2", try_multiply(col("X"), col("X")));

        df.show(20);
        df.printSchema();
        return df;
    }

    public Dataset<Row> AddXCubed(Dataset<Row> _df){

        Dataset<Row> df = null;

        df = _df.withColumn("X3", try_multiply(try_multiply(col("X"), col("X")), col("X")));

        df.show(20);
        df.printSchema();
        return df;
    }

    /**
     *
     * @param _df - dataframe to be processed
     * @param _order - order of the polynomial (ex: x^2, x^3)
     * @param _exclusive - do we want to use an order exclusively (so if it is x^3, we don't use x^2
     * @return processed dataset with the additional columns with the values of the x^order
     */
    public Dataset<Row> Process(Dataset<Row> _df, int _order, boolean _exclusive){

        VectorAssembler va = new VectorAssembler();
        String[] names = null;
        if(_order == 1){
            names = new String[]{"X"};
        }else if (_order == 2){
            names = new String[]{"X", "X2"};
        }
        else if(_order == 3 & !_exclusive){
            names = new String[]{"X", "X2", "X3"};
        }
        else if(_order == 3 & _exclusive){
            names = new String[]{"X", "X3"};
        }

        va.setInputCols(names);
        va.setOutputCol("features");
        var df = va.transform(_df);

        System.out.println("Processed to:");
        df.show(5);
        System.out.println("Processed dataframe's schema:");
        df.printSchema();

        return df;
    }

    public LinearRegressionModel LinearFit(Dataset<Row> _df_trans, int maxIter, double regParam, double elasticParam){
        LinearRegression lr = new LinearRegression()
                .setMaxIter(maxIter)
                .setRegParam(regParam)
                .setElasticNetParam(elasticParam)
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
}
