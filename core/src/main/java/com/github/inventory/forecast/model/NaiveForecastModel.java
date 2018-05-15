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

/**
 * <p>
 * Generates forecast for a sample by using the previous observation as the
 * next prediction.
 * </p>
 * <p>
 * For example, given the sample {@literal [11, 9, 13, 12, 11, 10]}, the
 * forecast will be {@literal [-, 11, 9, 13, 12, 11, 10]}, where {@literal "-"}
 * signifies that the prediction is unavailable or undefined.
 * </p>
 * <p>
 * This model simply says {@literal what happened last time will happen again}.
 * This strategy makes this model very cheap, as no computation. Given the way
 * this model generates predictions, it is mostly used as a baseline to compare
 * accuracy of other forecast models. It is best suited for scenarios where
 * the trend is mostly flat and there are no fluctuations in the sample data.
 * When the sample data fluctuates up or down, with some underlying trend,
 * this model may produce unusable predictions, since it does not take any
 * trends into account.
 * </p>
 */
public class NaiveForecastModel extends ForecastModel
{
  /**
   * {@inheritDoc}
   */
  @Override
  Forecast generateForecast(final double[] observations, final int projections)
  {
    final double[] predictions = new double[observations.length + projections];

    // Add an undefined prediction corresponding to the first observation
    // in the sample, as there is no precedent for the first observation.
    int i = 0;
    predictions[i++] = 0.0;

    double prediction = 0.0;

    for (final double observation : observations)
    {
      // Use the observed value as the prediction.
      predictions[i++] = prediction = observation;
    }

    // Add specified number of predictions beyond the sample.
    for (int j = 0; j < projections - 1; ++j)
    {
      // Add the prediction to the forecast.
      predictions[i++] = prediction;
    }

    return createForecast(observations, predictions);
  }
}
