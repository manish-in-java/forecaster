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

import java.util.Arrays;

/**
 * <p>
 * Represents a collection of observations. The order of observations in the
 * collection may not be chronological. If used with a forecasting model that
 * is sensitive to the chronological ordering of the observations, may produce
 * an inaccurate forecast if the observations are out of order.
 * </p>
 * <p>
 * This class is thread-safe.
 * </p>
 */
public class Sample
{
  private final double[] observations;

  /**
   * Constructs a sample of given observations.
   *
   * @param observations The observations for the sample.
   * @throws NullPointerException     if {@code observations} is
   *                                  {@literal null}.
   * @throws IllegalArgumentException if {@code observations} is empty.
   */
  public Sample(final double[] observations)
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

    this.observations = observations;
  }

  /**
   * Gets the observations included in the sample.
   *
   * @return A copy of the observations included in the sample. This ensures
   * that the actual sample, once generated, cannot be changed from outside.
   */
  public double[] getObservations()
  {
    return Arrays.copyOf(observations, observations.length);
  }
}
