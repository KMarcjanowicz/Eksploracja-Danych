package isi.ed;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.*;

public class LoadTags {
    public static Dataset<Row> Load() {
        SparkSession spark = SparkSession.builder()
                .appName("LoadTags")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "userId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "movieId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "tag",
                        DataTypes.StringType,
                        true),
                DataTypes.createStructField(
                        "timestamp",
                        DataTypes.StringType,
                        true),        });


        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("src/main/resources/tags.csv");

        df = df.withColumn("datetime", functions.from_unixtime(df.col("timestamp")))
                .withColumn("year", year(col("datetime")))
                .withColumn("month", month(col("datetime")))
                .withColumn("day", day(col("datetime")));

        System.out.println("Excerpt of the dataframe content:");

        df.show(5);
        System.out.println("Dataframe's schema:");
        df.printSchema();

        return df.groupBy(col("year"), col("month")).count().orderBy(col("year").desc(), col("month").desc());
    }

    public static Dataset<Row> JoinTags(){
        SparkSession spark = SparkSession.builder()
                .appName("LoadTags")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "userId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "movieId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "tag",
                        DataTypes.StringType,
                        true),
                DataTypes.createStructField(
                        "timestamp",
                        DataTypes.StringType,
                        true),        });


        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("src/main/resources/tags.csv");

        df = df.withColumn("datetime", functions.from_unixtime(df.col("timestamp")))
                .withColumn("year", year(col("datetime")))
                .withColumn("month", month(col("datetime")))
                .withColumn("day", day(col("datetime")));

        return df;
    }
}
