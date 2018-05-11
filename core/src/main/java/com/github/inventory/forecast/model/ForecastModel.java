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

import java.util.List;

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
   * @param projections The number of predictions to generate from the sample
   *                    beyond the observations already included in the sample.
   *                    If negative, is forcibly reset to {@literal zero}.
   *                    For example, if {@code sample} contains {@literal 10}
   *                    observations, and {@code projections} is specified as
   *                    {@literal 4}, the generated forecast will contain a
   *                    total of {@literal 14} predictions; one each for the
   *                    {@literal 10} observations in the sample, and
   *                    {@literal 4} additional beyond the sample set.
   * @return A {@link Forecast}, if {@code sample} is not {@literal null} or
   * empty, {@literal null} otherwise.
   */
  public Forecast forecast(final Sample sample, final int projections)
  {
    if (sample == null || sample.isEmpty())
    {
      return null;
    }

    return generateForecast(sample, Math.max(0, projections));
  }

  /**
   * Creates a forecast for a sample using specified predictions.
   *
   * @param sample      The sample for which the forecast has to be generated.
   * @param predictions The predictions to include in the forecast.
   * @return A {@link Forecast}.
   */
  Forecast createForecast(final Sample sample, final List<Double> predictions)
  {
    return new Forecast(sample, predictions);
  }

  /**
   * Generates a forecast based on a sample.
   *
   * @param sample      The sample based on which the forecast should be
   *                    generated.
   * @param projections The number of predictions to generate from the sample
   *                    beyond the observations already included in the sample.
   *                    If negative, is forcibly reset to {@literal zero}.
   *                    For example, if {@code sample} contains {@literal 10}
   *                    observations, and {@code projections} is specified as
   *                    {@literal 4}, the generated forecast will contain a
   *                    total of {@literal 14} predictions; one each for the
   *                    {@literal 10} observations in the sample, and
   *                    {@literal 4} additional beyond the sample set.
   * @return A {@link Forecast}.
   */
  abstract Forecast generateForecast(final Sample sample, final int projections);
}
