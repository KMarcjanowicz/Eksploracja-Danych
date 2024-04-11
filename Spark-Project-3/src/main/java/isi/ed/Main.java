package isi.ed;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.*;

import java.io.IOException;
import java.util.function.Function;

import static isi.ed.LinearRegressionPolynomialFeaturesPipeline.processDataset;
import static isi.ed.Plotting.plot;
import static isi.ed.Plotting.plot_stats_ym;
import static org.apache.spark.sql.functions.*;


public class Main {

    public static void main(String[] args) throws PythonExecutionException, IOException {

        SparkSession spark = SparkSession.builder()
                .appName("Functions")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        Function<Double, Double> f2 = (Double x) -> -1.5 * x*x + 3*x + 4;
        Function<Double, Double> f3 = (Double x) -> -1.5 * x*x + 3*x + 4;
        Function<Double, Double> f4 = (Double x) -> -10 * x*x + 500*x - 25;
        Function<Double, Double> f5 = (Double x) -> (x+4)*(x+1)*(x-3);

//        LinearRegressionPolynomialFeatures lrp = new LinearRegressionPolynomialFeatures();


//        df_2 = lrp.AddXSquared(df_2);
//        df_2 = lrp.AddXCubed(df_2);
//        df_2 = lrp.Process(df_2, 3, true);
//        LinearRegressionModel lrm_2 = lrp.LinearFit(df_2, 20,0.5, 0.8);
//
//        plot(
//                df_2.select(col("X")).as(Encoders.DOUBLE()).collectAsList(),
//                df_2.select(col("Y")).as(Encoders.DOUBLE()).collectAsList(),
//                lrm_2,
//                "Linear regression",
//                f2, 2);
//
//        Dataset<Row> df_4 = lrp.Load(4);
//        //df_4 = lrp.AddXSquared(df_4);
//        df_4 = lrp.AddXCubed(df_4);
//        df_4 = lrp.Process(df_4, 3, true);
//        LinearRegressionModel lrm_4 = lrp.LinearFit(df_4, 20, 0.5, 0.8);
//
//        plot(
//                df_4.select(col("X")).as(Encoders.DOUBLE()).collectAsList(),
//                df_4.select(col("Y")).as(Encoders.DOUBLE()).collectAsList(),
//                lrm_4,
//                "Linear regression",
//                f4, 2);
//
//        Dataset<Row> df_5 = lrp.Load(5);
//        //df_5 = lrp.AddXSquared(df_5);
//        df_5 = lrp.AddXCubed(df_5);
//        df_5 = lrp.Process(df_5, 3, true);
//        LinearRegressionModel lrm_5 = lrp.LinearFit(df_5, 20, 0.5, 0.8);
//
//        plot(
//                df_5.select(col("X")).as(Encoders.DOUBLE()).collectAsList(),
//                df_5.select(col("Y")).as(Encoders.DOUBLE()).collectAsList(),
//                lrm_5,
//                "Linear regression",
//                f5, 4);

        processDataset(spark, "xy-003.csv", 3, f3);
        processDataset(spark, "xy-005.csv", 2, f5);
    }
}