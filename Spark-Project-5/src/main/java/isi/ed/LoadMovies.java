package isi.ed;

import org.apache.spark.ml.tree.Split;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.*;

public class LoadMovies {
    public static Dataset<Row> Load(){
        SparkSession spark = SparkSession.builder()
                .appName("LoadMovies")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "movieId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "title",
                        DataTypes.StringType,
                        false),
                DataTypes.createStructField(
                        "genres",
                        DataTypes.StringType,
                        false),
        });



        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("src/main/resources/movies.csv");

        System.out.println("Excerpt of the dataframe content:");



        var df_transformed = df
                .withColumn("title2",regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)$", 1))
                .withColumn("year",regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)$", 2));
        df_transformed = df_transformed.drop("title");
        df_transformed = df_transformed.withColumnRenamed("title2","title");

        df_transformed = df_transformed.withColumn("genres_array", split(col("genres"), "\\|"));
        df_transformed = df_transformed.withColumn("genres2", explode(col("genres_array")));
        //df_transformed = df_transformed.drop("genres_array");
        df_transformed = df_transformed.drop("genres");
        df_transformed = df_transformed.withColumnRenamed("genres2","genres");


        df = df_transformed;

        df.show(20);
        System.out.println("Dataframe's schema:");
        df.printSchema();

        df.select("genres").distinct().show(false);

        var genreList = df.select("genres").distinct().as(Encoders.STRING()).collectAsList();
        for(var s:genreList){
            System.out.println(s);
        }
        System.out.println("\n");
        var df_multigenre = df_transformed;
        for(var s:genreList){
            if(s.equals("(no genres listed)"))continue;
            df_multigenre=df_multigenre.withColumn(s,array_contains(col("genres_array"),s));
            System.out.println(s);
        }

        return df;
    }

    public static Dataset<Row> LoadJoinMovies(){
        SparkSession spark = SparkSession.builder()
                .appName("LoadMovies")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField(
                        "movieId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "title",
                        DataTypes.StringType,
                        false),
                DataTypes.createStructField(
                        "genres",
                        DataTypes.StringType,
                        false),
        });



        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
                .schema(schema)
                .load("src/main/resources/movies.csv");

        System.out.println("Excerpt of the dataframe content:");

        df = df.withColumn("year",regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)\\s*.*$", 2))
                .withColumn("title2",
                        when(regexp_extract(col("title"),"^(.*?)\\s*\\((\\d{4})\\)\\s*.*$",1).equalTo("")
                                ,col("title"))
                                .otherwise(regexp_extract(col("title"),"^(.*?)\\s*\\((\\d{4})\\)\\s*.*$",1)))
                .drop("title")
                .withColumnRenamed("title2","title");



        return df;
    }

}
