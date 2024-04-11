package isi.ed;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.feature.PolynomialExpansion;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.function.Function;

import static isi.ed.Plotting.plot;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.try_multiply;

public class LinearRegressionPolynomialFeaturesPipeline {

    static void processDataset(SparkSession spark, String filename, int degree, Function<Double,Double> f_true){
        //import dataset from file
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
                .load("src/main/resources/" + filename);

        System.out.println("Excerpt of the " + filename + " content:");

        df.show(5);
        System.out.println(filename + " schema:");
        df.printSchema();

//        long rowsCount = df.count();
//        int trainCount = (int)(rowsCount*.7);
//        var df_train = df.select("*").limit(trainCount);
//        var df_test = df.select("*").offset(trainCount);
//        System.out.println(df_train.count());
//        System.out.println(df_test.count());

        df = df.orderBy(org.apache.spark.sql.functions.rand(3));
        var dfs = df.randomSplit(new double[]{0.7,0.3});
        var df_train = dfs[0];
        var df_test = dfs[1];

        String[] names = null;
        if(degree == 1){
            names = new String[]{"X"};
        }else if (degree == 2){
            names = new String[]{"X", "X2"};
            df_train = df_train.withColumn("X2", try_multiply(col("X"), col("X")));
            df_test = df_test.withColumn("X2", try_multiply(col("X"), col("X")));
        }
        else if(degree == 3){
            names = new String[]{"X", "X3"};
            df_train = df_train.withColumn("X3", try_multiply(try_multiply(col("X"), col("X")), col("X")));
            df_test = df_test.withColumn("X3", try_multiply(try_multiply(col("X"), col("X")), col("X")));
        }

        df_train.printSchema();
        df_test.printSchema();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(names)
                .setOutputCol("features");

        //Dataset<Row> df_trans = vectorAssembler.transform(df);

        PolynomialExpansion polyExpansion = new PolynomialExpansion()
                .setInputCol("features")
                .setOutputCol("polyFeatures")
                .setDegree(degree);

        //df_trans = polyExpansion.transform(df_trans);

        LinearRegression lr = new LinearRegression()
                .setMaxIter(30)
                .setRegParam(0.5)
                .setElasticNetParam(0.8)
                .setFeaturesCol("features")
                .setLabelCol("Y");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {vectorAssembler, polyExpansion, lr});
        PipelineModel model = pipeline.fit(df_train);

        var df_test_prediction = model.transform(df_test);
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("Y")
                .setPredictionCol("prediction")
                .setMetricName("rmse"); // or any other evaluation metric

        double rmse = evaluator.evaluate(df_test_prediction);
        evaluator.setMetricName("r2");
        double r2 = evaluator.evaluate(df_test_prediction);

        System.out.println("RMSE: " + rmse);
        System.out.println("R^2: " + r2);

        var x = df_test.select("X").as(Encoders.DOUBLE()).collectAsList();
        var y = df_test.select("Y").as(Encoders.DOUBLE()).collectAsList();
        plot(x,y,model,spark,String.format("Linear regression: %s (test data)",filename), degree, f_true);

//        LinearRegressionModel lrModel = (LinearRegressionModel)model.stages()[2];
    }
    
}
