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

/**
 * <p>
 * Generates a forecast for a sample as the average of successive smaller
 * groups of observations in the sample.
 * </p>
 *
 * <p>
 * For example, given the sample {@literal [11, 9, 13, 12, 11, 10]}, and
 * a group size of {@literal 4}, the forecast will be
 * {@literal [-, -, -, 11.25, 11.25, 11.5, 11.125]} (where {@literal "-"}
 * signifies that the prediction is unavailable or undefined), as explained
 * below.
 * </p>
 *
 * <ul>
 * <li>{@literal moving-average (11)               = -}</li>
 * <li>{@literal moving-average (11, 9)            = -}</li>
 * <li>{@literal moving-average (11, 9, 13)        = -}</li>
 * <li>{@literal moving-average (11, 9, 13, 12)    = 11.25}</li>
 * <li>{@literal moving-average (9, 13, 12, 11)    = 11.25}</li>
 * <li>{@literal moving-average (13, 12, 11, 10)   = 11.5}</li>
 * <li>{@literal moving-average (12, 11, 10, 11.5) = 11.125}</li>
 * </ul>
 *
 * <p>
 * The first three predictions are undefined because each group requires
 * {@literal 4} observations, but since there are fewer than the required
 * observations for the first three, the average cannot be calculated.
 * </p>
 *
 * <p>
 * This model is a special case of the {@link WeightedAverageForecastModel},
 * such that all observations in a group have equal weight. Due to this
 * reason, all advantages and disadvantages that apply to the later model
 * also apply to this model. Additionally, the shortcomings of the later
 * model are amplified here, because more recent observations are weighed
 * the same as older observations, which makes the model slower to respond
 * to changes in trends.
 * </p>
 *
 * @see <a href="https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc421.htm">Single Moving Average</a>
 */
public class SingleMovingAverageForecastModel extends WeightedAverageForecastModel
{
  /**
   * Creates a single moving average forecast model with a specified group
   * size, that is, the number of observations to consider as a group when
   * calculating the average.
   *
   * @param group The group size to use for the model. Forcibly set to
   *              {@literal 1} if {@literal 0} or negative. Using a group
   *              size that is greater than the number of observations in a
   *              sample will lead to undefined predictions, due to the fact
   *              that the minimum group required to calculate the moving
   *              average will fail to form.
   */
  public SingleMovingAverageForecastModel(final int group)
  {
    super();

    setWeights(Math.max(1, group));
  }

  /**
   * Sets weights to use for calculating the moving average using the
   * underlying weighted average model.
   *
   * @param group The group size to use for the model.
   */
  private void setWeights(final int group)
  {
    final double[] weights = new double[group];

    for (int i = 0; i < group; ++i)
    {
      weights[i] = 1;
    }

    setWeights(weights);
  }
}
