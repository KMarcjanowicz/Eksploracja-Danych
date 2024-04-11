package isi.ed;
import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static isi.ed.Plotting.plot;
import static org.apache.spark.sql.functions.*;


public class Main {

    public static void main(String[] args) throws PythonExecutionException, IOException {

        Function<Double, Double> f1 = (Double x) -> 2.37 * x + 7;
        Function<Double, Double> f2 = (Double x) -> -1.5 * x*x + 3*x + 4;
        Function<Double, Double> f4 = (Double x) -> -10 * x*x + 500*x - 25;
        //Dataset<Row> users = LoadUsers.Load();
        //Dataset<Row> movies = LoadMovies.Load();
        //Dataset<Row> ratings = LoadRatings.Load();
        //plot_stats_ym(ratings, "Ratings", "No. of Ratings across the years");

        //Dataset<Row> tags = LoadTags.Load();
        //plot_stats_ym(tags, "Tags", "No. of Tags across the years");

        //Dataset<Row> df_movies = LoadMovies.LoadJoinMovies();
        //Dataset<Row> df_ratings = LoadRatings.LoadJoinRatings();

        //JoinMoviesRatings(df_movies, df_ratings);

        //JoinMoviesRatingsGenres(LoadMovies.LoadJoinMovies(), LoadRatings.LoadJoinRatings());

        //JoinUserTags.JoinUserTags(LoadUsers.Load(), LoadTags.JoinTags());

        //JoinUserRatings.JoinUserRatings(LoadUsers.Load(), LoadRatings.LoadJoinRatings());

        LinearRegressionClass lr = new LinearRegressionClass();
//        Dataset<Row> df1 = lr.Load(1);
//        df1 = lr.Process(df1);
//        LinearRegressionModel lrModel = lr.LinearFit(df1, 0.3, 0.8);
//        plot(
//                df1.select(col("X")).as(Encoders.DOUBLE()).collectAsList(),
//                df1.select(col("Y")).as(Encoders.DOUBLE()).collectAsList(),
//                lrModel,
//                "Linear regression",
//                f1);

        Dataset<Row> df2 = lr.Load(2);
        df2 = lr.Process(df2);

        LinearRegressionModel lrModel2 = lr.LinearFit(df2, 20, 0.5 );
        plot(
                df2.select(col("X")).as(Encoders.DOUBLE()).collectAsList(),
                df2.select(col("Y")).as(Encoders.DOUBLE()).collectAsList(),
                lrModel2,
                "Linear regression",
                f2);

        Dataset<Row> df_ones = lr.OLS(df2);
//
//        Dataset<Row> df4 = lr.Load(4);
//        df4 = lr.Process(df4);
//
//        LinearRegressionModel lrModel4 = lr.LinearFit(df4, 20, 0.5);
//        plot(
//                df4.select(col("X")).as(Encoders.DOUBLE()).collectAsList(),
//                df4.select(col("Y")).as(Encoders.DOUBLE()).collectAsList(),
//                lrModel4,
//                "Linear regression",
//                f4);
    }
}