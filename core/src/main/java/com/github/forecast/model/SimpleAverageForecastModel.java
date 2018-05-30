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

import java.util.Arrays;

/**
 * <p>
 * Generates the forecast for a sample as the simple average of the observed
 * values.
 * </p>
 *
 * <p>
 * For example, given the sample {@literal [11, 9, 13, 12, 11, 10]}, the
 * forecast will be {@literal [11, 10, 11, 11.25, 11.2, 11, 11]}, as
 * explained below.
 * </p>
 *
 * <ul>
 * <li>{@literal average (11)                        = 11}</li>
 * <li>{@literal average (11, 9)                     = 10}</li>
 * <li>{@literal average (11, 9, 13)                 = 11}</li>
 * <li>{@literal average (11, 9, 13, 12)             = 11.25}</li>
 * <li>{@literal average (11, 9, 13, 12, 11)         = 11.2}</li>
 * <li>{@literal average (11, 9, 13, 12, 11, 10)     = 11}</li>
 * <li>{@literal average (11, 9, 13, 12, 11, 10, 11) = 11}</li>
 * </ul>
 *
 * <p>
 * This model is very simple and can make predictions quickly. It does not
 * require storing large amounts of data to generate a prediction, and
 * amplifies the most recent trends (upward or downward movement).
 * </p>
 *
 * <p>
 * On the flip-side, a simple average forecast settles around the average
 * for a large sample size, thereby deleting any trends hidden within the
 * sample. For this reason, this model does not take into account any
 * trends at all for long-term forecasts.
 * </p>
 *
 * @see <a href="https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc42.htm">Averaging Techniques</a>
 */
public class SimpleAverageForecastModel extends ForecastModel
{
  /**
   * {@inheritDoc}
   */
  @Override
  Forecast forecast(final double[] observations, final int projections)
  {
    final int samples = observations.length;

    final double[] predictions = new double[samples + projections];

    double prediction = 0.0;

    // Generate predictions for each observation.
    for (int i = 0; i < samples; ++i)
    {
      // Add the simple average for the observations encountered so far
      // to the forecast.
      predictions[i] = prediction = simpleAverage(Arrays.copyOf(observations, i + 1));
    }

    // Add specified number of predictions beyond the sample.
    Arrays.fill(predictions, samples, samples + projections, prediction);

    return forecast(observations, predictions);
  }
}
