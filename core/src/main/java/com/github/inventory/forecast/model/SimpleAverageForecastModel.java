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
import com.github.inventory.forecast.domain.Sample;

import java.util.ArrayList;
import java.util.List;

/**
 * <p>
 * Generates the forecast for a sample as the simple average of the observed
 * values.
 * </p>
 * <p>
 * For example, given the sample {@literal [11, 9, 13, 12, 11, 10]}, the
 * forecast will be {@literal [11, 10, 11, 11.25, 11.2, 11, 11]}, as
 * explained below.
 * </p>
 * <ul>
 * <li>{@literal average (11)                        = 11}</li>
 * <li>{@literal average (11, 9)                     = 10}</li>
 * <li>{@literal average (11, 9, 13)                 = 11}</li>
 * <li>{@literal average (11, 9, 13, 12)             = 11.25}</li>
 * <li>{@literal average (11, 9, 13, 12, 11)         = 11.2}</li>
 * <li>{@literal average (11, 9, 13, 12, 11, 10)     = 11}</li>
 * <li>{@literal average (11, 9, 13, 12, 11, 10, 11) = 11}</li>
 * </ul>
 * <p>
 * The bias for all predictions made by this model is {@literal zero}, which
 * is due to the fact that the bias is defined as the average total error and
 * the model predicts the average as the forecast, cancelling the two out.
 * </p>
 * <p>
 * This model is very simple and can make predictions quickly. It does not
 * require storing large amounts of data to generate a prediction, and
 * amplifies the most recent trends (upward or downward movement).
 * </p>
 * <p>
 * On the flip-side, a simple average forecast settles around the average
 * for a large sample size, thereby deleting any trends hidden within the
 * sample. For this reason, this model does not take into account seasonal,
 * or time-based trends.
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
  Forecast generateForecast(final Sample sample, final int predictions)
  {
    final List<Double> forecast = new ArrayList<>(sample.size() + predictions);

    final List<Double> observations = new ArrayList<>();
    double prediction = 0;

    // Generate predictions for each observation.
    for (final Double observation : sample)
    {
      observations.add(observation);

      // Find the simple average for the observations.
      prediction = observations.stream()
                               .mapToDouble(d -> d)
                               .average()
                               .orElse(0);

      // Add the prediction to the forecast.
      forecast.add(prediction);
    }

    // Add specified number of predictions beyond the sample.
    for (int i = 0; i < predictions; ++i)
    {
      observations.add(prediction);

      // Find the simple average for the observations.
      prediction = observations.stream()
                               .mapToDouble(d -> d)
                               .average()
                               .orElse(0);

      // Add the prediction to the forecast.
      forecast.add(prediction);
    }

    return createForecast(sample, forecast);
  }
}
