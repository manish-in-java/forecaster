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

import java.util.Arrays;

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
{
  private       double   bias;
  private       double   meanAbsoluteDeviation;
  private       double   meanAbsolutePercentageError;
  private       double   meanSquaredError;
  private final double[] predictions;
  private       double   totalAbsoluteError;
  private       double   totalSquaredError;

  /**
   * Constructs a forecast for a given sample of observations and associated
   * predictions.
   *
   * @param observations The sample for which the forecast has been generated.
   * @param predictions  The predictions to include in the forecast.
   * @throws NullPointerException     if {@code observations} or
   *                                  {@code predictions} is {@literal null}.
   * @throws IllegalArgumentException if  {@code observations} or
   *                                  *                                  {@code predictions} is empty.
   */
  public Forecast(final double[] observations, final double[] predictions)
  {
    super();

    // Ensure that the observations have been specified.
    if (observations == null)
    {
      throw new NullPointerException("Observations must not be null.");
    }
    // Ensure that the observations are non-empty.
    else if (observations.length == 0)
    {
      throw new IllegalArgumentException("Observations must not be empty.");
    }

    // Ensure that the predictions have been specified.
    if (predictions == null)
    {
      throw new NullPointerException("Predictions must not be null.");
    }
    // Ensure that the predictions are non-empty.
    else if (predictions.length == 0)
    {
      throw new IllegalArgumentException("Predictions must not be empty.");
    }

    this.predictions = predictions;

    calculateMeasures(observations, predictions);
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
   * Gets the predictions included in the forecast.
   *
   * @return A copy of the predictions included in the forecast. This ensures
   * that the actual forecast, once generated, cannot be changed from outside.
   */
  public double[] getPredictions()
  {
    return Arrays.copyOf(predictions, predictions.length);
  }

  /**
   * Gets the {@literal total-absolute-error} (also known as
   * {@literal sum-of-absolute-error (SAE)}) for the forecast.
   *
   * @return The {@literal total-absolute-error} for the forecast.
   */
  public double getTotalAbsoluteError()
  {
    return totalAbsoluteError;
  }

  /**
   * Gets the {@literal total-squared-error} (also known as
   * {@literal sum-of-squared-error (SSE)}) for the forecast.
   *
   * @return The {@literal total-squared-error} for the forecast.
   */
  public double getTotalSquaredError()
  {
    return totalSquaredError;
  }

  /**
   * Calculates and saves measures of accuracy for the predicted values
   * versus the observed values.
   *
   * @param observations The observed values based on which the forecast
   *                     was generated.
   * @param predictions  The predicted values for the forecast.
   */
  private void calculateMeasures(final double[] observations, final double[] predictions)
  {
    double absoluteError = 0.0, absolutePercentageError = 0.0, squaredError = 0.0, totalError = 0.0;

    // Calculate errors for the observed values.
    final int observationCount = observations.length;
    for (int i = 0; i < observationCount; ++i)
    {
      final double observation = observations[i], prediction = predictions[i];

      // Determine the error in prediction. If the predicted value is zero,
      // consider it to be invalid and use the observed value itself as the
      // predicted value.
      final double error = observation - (prediction != 0 ? prediction : observation);

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
    this.totalSquaredError = squaredError;
  }
}
