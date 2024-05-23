package isi.ed;

import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;

import java.io.IOException;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.apache.spark.sql.functions.*;

public class Plotting {
    static void plot_stats_ym(Dataset<Row> df, String title, String label) {
        var labels = df.select(concat(col("year"), lit("-"), col("month"))).as(Encoders.STRING()).collectAsList();
        var x = NumpyUtils.arange(0, labels.size() - 1, 1);
        x = df.select(expr("year+(month-1)/12")).as(Encoders.DOUBLE()).collectAsList();
        var y = df.select("count").as(Encoders.DOUBLE()).collectAsList();
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        plt.plot().add(x, y).linestyle("-").label(label);
        plt.legend();
        plt.title(title);
        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    static void plot_histogram(List<Double> x, String title) {
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        plt.hist().add(x).bins(50);
        plt.title(title);
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    static void plot_histogram(List<Double> x, List<Double> weights, String title) {
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        plt.hist().add(x).weights(weights).bins(50);
        plt.title(title);
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    static void plot_scatter(List<Double> x, List<Double> y, String title){
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        plt.plot().add(x, y, "o").label("data");
        plt.title(title);
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    static void plotObjectiveHistory(List<Double> lossHistory){
        var x = IntStream.range(0,lossHistory.size()).mapToDouble(d->d).boxed().toList();
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        plt.plot().add(x, lossHistory).label("loss");
        plt.xlabel("Iteration");
        plt.ylabel("Loss");
        plt.title("Loss history");
        plt.legend();
        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @param x - współrzedne x danych
     * @param y - współrzedne y danych
     * @param lrModel - model regresji
     * @param title - tytuł do wyswietlenia (może być null)
     * @param f_true - funkcja f_true (może być null)
     */
    static void plot(List<Double> x, List<Double> y, LinearRegressionModel lrModel, String title, Function<Double,Double> f_true){
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        plt.plot().add(x, y,"o").color("b").label("data");
        plt.xlabel("X: ");
        plt.ylabel("Y: ");

        double xmin = 0, xmax = 0;

        if(x.stream().min(Double::compare).isPresent()) xmin = x.stream().min(Double::compare).get();
        if(x.stream().max(Double::compare).isPresent()) xmax = x.stream().max(Double::compare).get();

        var xdelta = 0.05*(xmax-xmin);
        List<Double> fx = NumpyUtils.linspace(xmin-xdelta,xmax+xdelta,100);

        double[] fx_arr = fx.stream().mapToDouble(Double::doubleValue).toArray(); //via method reference

        List<Double> fy = fx.stream().map((it) -> lrModel.predict(new DenseVector(new double[]{it}))).collect(Collectors.toList());

        plt.plot().add(fx, fy).color("r").label("prediction");

        if(f_true != null){
            List<Double> fy_true = fx.stream().map(f_true).collect(Collectors.toList());
            plt.plot().add(fx, fy_true).color("g").linestyle("--").label("$f_{true}$");
        }

        if(title != null) plt.title(title);
        plt.legend();

        try {
            plt.show();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }
}
