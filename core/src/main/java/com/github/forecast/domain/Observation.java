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

package com.github.forecast.domain;

import java.io.Serializable;

/**
 * Represents a single data point to be used for forecasting.
 */
public class Observation implements Serializable
{
  private final double value;

  /**
   * Creates an observation with a given value.
   *
   * @param value The observed value.
   */
  public Observation(final double value)
  {
    this.value = value;
  }

  /**
   * Gets the observed value.
   *
   * @return The observed value.
   */
  public double getValue()
  {
    return value;
  }
}
