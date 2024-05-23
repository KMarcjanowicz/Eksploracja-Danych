package isi.ed;

import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import static org.apache.spark.sql.functions.*;

public class LoadRatings {

    public static Dataset<Row> Load() throws PythonExecutionException, IOException {
        SparkSession spark = SparkSession.builder()
                .appName("LoadRatings")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(
                        "userId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "movieId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "rating",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "timestamp",
                        DataTypes.IntegerType,
                        true),
        });


        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("src/main/resources/ratings.csv");

        df = df.withColumn("datetime", functions.from_unixtime(df.col("timestamp")))
                .withColumn("year", substring(col("datetime"), 0, 4))
                .withColumn("month", substring(col("datetime"), 6, 2))
                .withColumn("day", substring(col("datetime"), 9, 2));


        System.out.println("Excerpt of the dataframe content:");
        df.show(5);
        System.out.println("Dataframe's schema:");
        df.printSchema();

        return df.groupBy(col("year"), col("month")).count().orderBy(col("year").desc(), col("month").desc());
    }

    public static Dataset<Row> LoadJoinRatings(){
        SparkSession spark = SparkSession.builder()
                .appName("LoadRatings")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(
                        "userId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "movieId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "rating",
                        DataTypes.DoubleType,
                        true),
                DataTypes.createStructField(
                        "timestamp",
                        DataTypes.IntegerType,
                        true),
        });


        return spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("src/main/resources/ratings.csv");
    }
}
