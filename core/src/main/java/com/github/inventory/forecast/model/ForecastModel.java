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

package com.github.inventory.forecast.model;

import com.github.inventory.forecast.domain.Forecast;
import com.github.inventory.forecast.domain.Prediction;
import com.github.inventory.forecast.domain.Sample;

import java.util.Collection;

/**
 * Contract for generating a forecast based on a sample.
 */
public abstract class ForecastModel
{
  /**
   * Generates a forecast based on a sample, containing a single prediction
   * beyond the sample size.
   *
   * @param sample The sample based on which the forecast should be generated.
   * @return A {@link Forecast}, if {@code sample} is not {@literal null} or
   * empty, {@literal null} otherwise.
   */
  public Forecast forecast(final Sample sample)
  {
    return forecast(sample, 1);
  }

  /**
   * Generates a forecast based on a sample, containing a specified number of
   * predictions beyond the sample size.
   *
   * @param sample      The sample based on which the forecast should be
   *                    generated.
   * @param predictions The number of predictions to generate from the sample.
   *                    The actual number of predictions generated and included
   *                    in the forecast depends on the actual forecast model
   *                    used. If the specified number of predictions is
   *                    negative, it is forcibly reset to {@literal zero}.
   * @return A {@link Forecast}, if {@code sample} is not {@literal null} or
   * empty, {@literal null} otherwise.
   */
  public Forecast forecast(final Sample sample, final int predictions)
  {
    if (sample == null || sample.isEmpty())
    {
      return null;
    }

    return generateForecast(sample, Math.max(0, predictions));
  }

  /**
   * Gets the prediction for a sample, given a forecast value.
   *
   * @param sample   The sample for which the forecast has been generated.
   * @param forecast The forecast value for the sample.
   * @return A {@link Prediction}.
   */
  Prediction getPrediction(final Collection<Double> sample, final double forecast)
  {
    double absoluteError = 0.0, absolutePercentageError = 0.0, squaredError = 0.0, totalError = 0.0;

    for (final Double observation : sample)
    {
      final double error = forecast - observation;

      absoluteError += Math.abs(error);
      absolutePercentageError += Math.abs(error / observation);
      squaredError += error * error;
      totalError += error;
    }

    final int observations = sample.size();

    return new Prediction(forecast
        , absoluteError
        , totalError / observations
        , squaredError / observations
        , absoluteError / observations
        , absolutePercentageError / observations);
  }

  /**
   * Generates a forecast based on a sample.
   *
   * @param sample      The sample based on which the forecast should be generated.
   * @param predictions The number of predictions to generate from the sample.
   * @return A {@link Forecast}.
   */
  abstract Forecast generateForecast(final Sample sample, final int predictions);
}
