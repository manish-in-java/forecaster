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
import com.github.inventory.forecast.domain.Observation;
import com.github.inventory.forecast.domain.Prediction;
import com.github.inventory.forecast.domain.Sample;

import java.util.ArrayList;
import java.util.List;

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
  Forecast generateForecast(final Sample sample, final int predictions)
  {
    final Forecast forecast = new Forecast(sample.size() + predictions);

    // Add an undefined prediction corresponding to the first observation
    // in the sample, as there is no precedent for the first observation.
    forecast.add(Prediction.undefined());

    final List<Double> observations = new ArrayList<>();
    double prediction = 0;

    for (final Observation observation : sample)
    {
      observations.add(observation.getValue());

      // Use the observed value as the prediction.
      prediction = observation.getValue();

      // Copy the observed value as the next prediction.
      forecast.add(getPrediction(observations, prediction));
    }

    // Add specified number of predictions beyond the sample.
    for (int i = 0; i < predictions - 1; ++i)
    {
      observations.add(prediction);

      // Add the prediction to the forecast.
      forecast.add(getPrediction(observations, prediction));
    }

    return forecast;
  }
}
