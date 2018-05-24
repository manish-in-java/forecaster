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

package com.github.forecast.model;

import com.github.forecast.domain.Forecast;

/**
 * <p>
 * Generates the forecast for a sample as the simple average of all the
 * observed values.
 * </p>
 *
 * <p>
 * This model uses the same value for all predictions (since the average of
 * any set of values is always unique). Therefore, it is not very useful
 * from a practical standpoint and should be used only for demonstration
 * purposes.
 * </p>
 */
public class StraightAverageForecastModel extends ForecastModel
{
  /**
   * {@inheritDoc}
   */
  @Override
  Forecast forecast(final double[] observations, final int projections)
  {
    final double[] predictions = new double[observations.length + projections];

    // Find the simple average for the observations.
    final double average = simpleAverage(observations);

    // Add predictions for each observation.
    for (int i = 0; i < observations.length + projections; ++i)
    {
      predictions[i] = average;
    }

    return forecast(observations, predictions);
  }
}
