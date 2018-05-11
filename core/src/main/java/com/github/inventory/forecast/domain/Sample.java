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

import java.util.Collection;
import java.util.List;
import java.util.Vector;

/**
 * <p>
 * Represents a collection of observations. The order of the observations
 * in the collection may not be chronological. If used with a forecasting
 * model that is sensitive to the chronological ordering of the observations,
 * may produce an inaccurate forecast.
 * </p>
 * <p>
 * This class is thread-safe.
 * </p>
 */
public class Sample
    extends Vector<Double>
    implements List<Double>, Collection<Double>
{
  /**
   * Constructs an empty sample.
   */
  public Sample()
  {
    super();
  }

  /**
   * Constructs an empty sample with capacity to hold the specified number of
   * observations.
   *
   * @param size The number of observations in the sample.
   * @throws IllegalArgumentException if the number of observations is
   *                                  negative.
   */
  public Sample(final int size)
  {
    super(size);
  }
}
