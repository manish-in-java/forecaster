/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.inventory.forecast.domain;

import com.github.inventory.forecast.model.ForecastModel;

import java.util.Collection;
import java.util.List;
import java.util.Vector;

/**
 * <p>
 * Represents a collection of predictions. A forecast is generated on a
 * {@link Sample} of observations by applying a {@link ForecastModel}.
 * </p>
 * <p>
 * The applied {@link ForecastModel} generates a prediction for every
 * single observation in the {@link Sample}. Since observations are
 * data points collected in reality, the prediction generated for an
 * observation allows comparing the actual value versus the predicted
 * value. In addition to predictions for each observation, forecast models
 * also generate a specified number of additional predictions.
 * </p>
 * <p>
 * Each forecast also contains a number of measures that allow the forecast
 * accuracy to be expressed quantitatively. These measures show how much the
 * predictions vary from the actual observations in the sample, and can help
 * determine the relative efficacy of different forecast models with the same
 * sample data.
 * </p>
 * <p>
 * This class is thread-safe.
 * </p>
 */
public class Forecast
    extends Vector<Double>
    implements List<Double>, Collection<Double>
{
  private double bias;
  private double meanAbsoluteDeviation;
  private double meanAbsolutePercentageError;
  private double meanSquaredError;
  private double totalAbsoluteError;

  /**
   * Constructs a forecast for a given sample and its associated predictions.
   *
   * @param sample      The sample for which the forecast has been generated.
   * @param predictions The predictions to include in the forecast.
   * @throws NullPointerException if {@code predictions} is {@literal null}.
   */
  public Forecast(final Sample sample, final List<Double> predictions)
  {
    super(predictions);

    calculateMeasures(sample, predictions);
  }

  /**
   * Gets the bias for the forecast, which is the average error.
   *
   * @return The bias for the forecast.
   */
  public double getBias()
  {
    return bias;
  }

  /**
   * Gets the {@literal mean-absolute-deviation} ({@literal MAD}) for the
   * forecast.
   *
   * @return The {@literal mean-absolute-deviation} for the forecast.
   * @see <a href="https://en.wikipedia.org/wiki/Mean_absolute_deviation">Mean Absolute Deviation</a>
   */
  public double getMeanAbsoluteDeviation()
  {
    return meanAbsoluteDeviation;
  }

  /**
   * Gets the {@literal mean-absolute-percentage-error} ({@literal MAPE})
   * for the forecast.
   *
   * @return The {@literal mean-absolute-percentage-error} for the forecast.
   * @see <a href="https://en.wikipedia.org/wiki/Mean_absolute_percentage_error">Mean Absolute Percentage Error</a>
   */
  public double getMeanAbsolutePercentageError()
  {
    return meanAbsolutePercentageError;
  }

  /**
   * Gets the {@literal mean-squared-error} ({@literal MSE}) for the
   * forecast.
   *
   * @return The {@literal mean-squared-error} for the forecast.
   * @see <a href="https://en.wikipedia.org/wiki/Mean_squared_error">Mean Squared Error</a>
   */
  public double getMeanSquaredError()
  {
    return meanSquaredError;
  }

  /**
   * Gets the {@literal total-absolute-error} ({@literal SAE}) for the
   * forecast.
   *
   * @return The {@literal total-absolute-error} for the forecast.
   */
  public double getTotalAbsoluteError()
  {
    return totalAbsoluteError;
  }

  /**
   * Calculates and saves measures of accuracy for the predicted values
   * versus the observed values.
   *
   * @param observations The observed values based on which the forecast
   *                     was generated.
   * @param predictions  The predicted values for the forecast.
   */
  private void calculateMeasures(final List<Double> observations, final List<Double> predictions)
  {
    if (observations == null || observations.isEmpty() || predictions == null || predictions.isEmpty())
    {
      return;
    }

    double absoluteError = 0.0, absolutePercentageError = 0.0, squaredError = 0.0, totalError = 0.0;

    // Calculate errors for the observed values.
    final int observationCount = observations.size();
    for (int i = 0; i < observationCount; ++i)
    {
      final double observation = observations.get(i), prediction = predictions.get(i);

      // Determine the error in prediction. If the predicted value is zero,
      // consider it to be invalid and use the observed value itself as the
      // predicted value.
      final double error = (prediction != 0 ? prediction : observation) - observation;

      absoluteError += Math.abs(error);
      absolutePercentageError += Math.abs(error / observation);
      squaredError += error * error;
      totalError += error;
    }

    // Calculate and save accuracy measures.
    this.bias = totalError / observationCount;
    this.meanAbsoluteDeviation = absoluteError / observationCount;
    this.meanAbsolutePercentageError = absolutePercentageError / observationCount;
    this.meanSquaredError = squaredError / observationCount;
    this.totalAbsoluteError = absoluteError;
  }
}
