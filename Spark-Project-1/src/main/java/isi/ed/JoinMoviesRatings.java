package isi.ed;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

import static isi.ed.Plotting.plot_histogram;
import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;

public class JoinMoviesRatings {
    static private void JoinMoviesRatings(Dataset<Row> movies, Dataset<Row> ratings){

        //var df_mr = df_movies.join(df_ratings,df_movies.col("movieId").equalTo(df_ratings.col("movieId")));
        //Jeżeli nazwy kolumn są identyczne, można użyć:
        var df_mr = movies.join(ratings,"movieId","inner");
        df_mr = df_mr.groupBy("title").agg(
                min("rating").alias("min_rating"),
                avg("rating").alias("avg_rating"),
                max("rating").alias("max_rating"),
                count("rating").alias("cnt_rating")
        ).orderBy(col("cnt_rating").desc());
        System.out.println("Excerpt of the dataframe content:");
        df_mr.show(5);
        System.out.println("Dataframe's schema:");
        df_mr.printSchema();

        var avgRatings = df_mr.select("avg_rating").where("avg_rating>=0").as(Encoders.DOUBLE()).collectAsList();
        //plot_histogram(avgRatings, "Average Ratings");

        avgRatings = df_mr.select("avg_rating").where("avg_rating>=4.5").as(Encoders.DOUBLE()).collectAsList();
        //plot_histogram(avgRatings, "Ratings above 4.5");

        avgRatings = df_mr.select("avg_rating").where("avg_rating>=3.5 and avg_rating<=4.5 and cnt_rating>200").as(Encoders.DOUBLE()).collectAsList();
        //plot_histogram(avgRatings, "Ratings between 3.5 and 4.5 and where >200");

        var df_mr_t = movies.join(ratings,"movieId","inner");

        df_mr_t = df_mr_t.withColumn("datetime", functions.from_unixtime(df_mr_t.col("timestamp")))
                .drop(col("timestamp"))
                .withColumn("release_to_rating_year", functions.year(functions.col("datetime")).minus(functions.col("year")));

        System.out.println("release_to_rating_year:");
        df_mr_t.show(5);
        System.out.println("Dataframe's schema:");
        df_mr_t.printSchema();

        //down sampling
        var release_to_rating_year = df_mr_t.select("release_to_rating_year").sample((double) 500 / df_mr_t.count()).as(Encoders.DOUBLE()).collectAsList();
        //plot_histogram(release_to_rating_year, "500 samples: release_to_rating_year");

        df_mr_t = df_mr_t.groupBy(col("release_to_rating_year")).count().orderBy("release_to_rating_year");

        System.out.println("Zgrupuj dane po kolumnie release_to_rating_year:");
        df_mr_t.show(5);
        System.out.println("Dataframe's schema:");
        df_mr_t.printSchema();

        var df_mr_t2 = df_mr_t.filter("release_to_rating_year=-1 OR release_to_rating_year IS NULL");
        df_mr_t2.show(105);

        df_mr_t = df_mr_t.filter("NOT (release_to_rating_year=-1 OR release_to_rating_year IS NULL)");

        plot_histogram(
                df_mr_t.select("release_to_rating_year").as(Encoders.DOUBLE()).collectAsList(),
                df_mr_t.select("count").as(Encoders.DOUBLE()).collectAsList(),
                "Rozklad roznicy lat pomiedzy ocena a wydaniem filmu"
        );
    }
}
