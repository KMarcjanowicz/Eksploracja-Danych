package isi.ed;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.*;

import static org.apache.spark.sql.functions.*;

public class JoinMoviesGenres {
    public static void JoinMoviesRatingsGenres(Dataset<Row> movies, Dataset<Row> ratings){
        var df_mrg = movies.join(ratings,movies.col("movieId").equalTo(ratings.col("movieId")));
        df_mrg = df_mrg.drop(ratings.col("movieId"));
        df_mrg = df_mrg.withColumn("genres_array", split(col("genres"), "\\|"));
        df_mrg = df_mrg.withColumn("genres2", explode(col("genres_array")));
        df_mrg = df_mrg.drop("genres_array");
        df_mrg = df_mrg.drop("genres");
        df_mrg = df_mrg.withColumnRenamed("genres2","genres");

        var df_ratings = df_mrg.groupBy(col("genres")).agg(
                min("rating").alias("min_rating"),
                avg("rating").alias("avg_rating"),
                max("rating").alias("max_rating"),
                count("rating").alias("cnt_rating")
        ).orderBy(col("avg_rating").desc());

        //.limit(3)

        // global avarage
        double avg = ratings.agg(
                avg("rating")
        ).head().getDouble(0);

        var df_avg = df_ratings.filter("avg_rating >" + avg).orderBy(col("avg_rating").asc());

        //Spark session for the View logic
        SparkSession spark = SparkSession.builder()
                .appName("JoinMoviesGenres")
                .master("local")
                .getOrCreate();

        df_mrg.createOrReplaceTempView("movies_ratings");
        ratings.createOrReplaceTempView("ratings");
        String query = """
           SELECT genres, AVG(rating) AS avg_rating, COUNT(rating) 
           FROM movies_ratings GROUP BY genres 
           HAVING AVG(rating) > (SELECT AVG(rating) FROM ratings) 
           ORDER BY avg_rating DESC""";
        Dataset<Row> df_cat_above_avg = spark.sql(query);
        df_cat_above_avg.show();

//        System.out.println("Joined Datasets:");
//        df_avg.show(20);
//        System.out.println("Dataframe's schema:");
//        df_avg.printSchema();
    }
}
