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

import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Collectors;

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
   * @return A {@link Forecast}, if {@code sample} is not {@literal null},
   * {@literal null} otherwise.
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
    if (sample == null)
    {
      return null;
    }

    return forecast(sample.getObservations(), Math.max(0, projections));
  }

  /**
   * Creates a forecast for a sample of observations with corresponding
   * predictions.
   *
   * @param observations The observations for the sample for which the
   *                     forecast has to be created.
   * @param predictions  The predictions to include in the forecast.
   * @return A {@link Forecast}.
   */
  Forecast forecast(final double[] observations, final double[] predictions)
  {
    return new Forecast(observations, predictions);
  }

  /**
   * Generates a forecast based on a sample of observations.
   *
   * @param observations The observations based on which the forecast should be
   *                     generated.
   * @param projections  The number of predictions to generate from the sample
   *                     beyond the observations already included in the
   *                     sample. If negative, is forcibly reset to
   *                     {@literal zero}. For example, if there are
   *                     {@literal 10} observations, and {@code projections} is
   *                     specified as {@literal 4}, the generated forecast will
   *                     contain a total of {@literal 14} predictions; one each
   *                     for the {@literal 10} observations in the sample, and
   *                     {@literal 4} additional beyond the sample set.
   * @return A {@link Forecast}.
   */
  abstract Forecast forecast(final double[] observations, final int projections);

  /**
   * Gets the simple average for an array of observations.
   *
   * @param observations The array of observations for which the
   *                     simple average is required.
   * @return The simple average for the observations.
   */
  double simpleAverage(final double[] observations)
  {
    return simpleAverage(Arrays.stream(observations).boxed().collect(Collectors.toList()));
  }

  /**
   * Gets the simple average for a collection of observations.
   *
   * @param observations The collection of observations for which the
   *                     simple average is required.
   * @return The simple average for the observations.
   */
  private double simpleAverage(final Collection<Double> observations)
  {
    return observations.stream()
                       .mapToDouble(d -> d)
                       .average()
                       .orElse(0);
  }
}
