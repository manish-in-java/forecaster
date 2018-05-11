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

/**
 * <p>
 * Generates forecast for a sample as the weighted average of a given observed
 * value and the values of some number of its preceding values.
 * </p>
 * <p>
 * For example, given the sample {@literal [11, 9, 13, 12, 11, 10]}, and
 * weights {@literal [0.2, 0.4, 0.4]}, the forecast will be
 * {@literal [-, -, 11, 11.8, 11.8, 10.8, 10.52]} (where {@literal "-"}
 * signifies that the prediction is unavailable or undefined), as explained
 * below.
 * </p>
 * <ul>
 * <li>{@literal weighted-average (11)           = -}</li>
 * <li>{@literal weighted-average (11, 9)        = -}</li>
 * <li>{@literal weighted-average (11, 9, 13)    = 11*0.2 + 9*0.4 + 13*0.4    = 11}</li>
 * <li>{@literal weighted-average (9, 13, 12)    = 9*0.2 + 13*0.4 + 12*0.4    = 11.8}</li>
 * <li>{@literal weighted-average (13, 12, 11)   = 13*0.2 + 12*0.4 + 11*0.4   = 11.8}</li>
 * <li>{@literal weighted-average (12, 11, 10)   = 12*0.2 + 11*0.4 + 10*0.4   = 10.8}</li>
 * <li>{@literal weighted-average (11, 10, 10.8) = 11*0.2 + 10*0.4 + 10.8*0.4 = 10.52}</li>
 * </ul>
 * <p>
 * The first two predictions are undefined because there are three weights,
 * which require three observations to calculate the weighted average, but
 * since there are fewer than the required three observations for the first
 * two, the weighted average cannot be calculated.
 * </p>
 * <p>
 * Given that more recent observations should in general be a better
 * indication of future trends, a weighted-average forecast generated
 * using higher weights assigned to more recent observations is usually
 * more accurate than a simple-average forecast.
 * </p>
 * <p>
 * Other advantages of the model include suppression of random fluctuations
 * that could otherwise lead to much higher inventory holding costs, and
 * amplification of long-term trends.
 * </p>
 * <p>
 * The disadvantages of the weighted average model are that a minimum number
 * of observations (equal to the number of weights) must be available for a
 * forecast to be generated, and it is computationally more intensive than the
 * simple average model.
 * </p>
 * <p>
 * A major disadvantage of the model, given that it suppresses fluctuations,
 * is that it disregards anticipated fluctuations, on account of say,
 * seasonality. Therefore, forecasts for seasonal products may come out to be
 * highly inaccurate.
 * </p>
 *
 * @see <a href="https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc42.htm">Averaging Techniques</a>
 */
public class WeightedAverageForecastModel extends ForecastModel
{
  private double[] weights;

  /**
   * <p>
   * Creates a new weighted average forecast model using the specified weights.
   * </p>
   * <p>
   * The number of weights provided determines the number of observations from
   * the sample over which the weighted average needs to be computed. For
   * example, if four weights are provided, the weighted average is computed
   * over groups of four observations at a time.
   * </p>
   * <p>
   * Weights should be provided in descending order of precedence. This means
   * the first weight will be assigned to the most recent observation, the
   * second weight to the immediately preceding observation, and so on.
   * </p>
   * <p>
   * The number of weights affects the number of predictions that can be made
   * for a given sample of observations. For example, if five weights are used
   * to attempt a forecast on a sample of two observations, no predictions can
   * be made because the sample size is insufficient for the weighted average
   * to be calculated. For this reason, the model works best when the weights
   * are assigned judiciously, based on the sample size.
   * </p>
   * <p>
   * Although standard descriptions of the weighted average model use weights
   * adding up to {@literal one (1)}, the same restriction is not applicable
   * here. The assigned weights are automatically flattened out in the ratio
   * determined by the individual weights.
   * </p>
   *
   * @param weights The weights to use for computing the weighted average
   *                for observed values. If {@literal null} or empty, defaults
   *                to the simple average model.
   */
  public WeightedAverageForecastModel(final double[] weights)
  {
    super();

    setWeights(weights);
  }

  /**
   * {@inheritDoc}
   */
  @Override
  Forecast generateForecast(final Sample sample, final int predictions)
  {
    final Forecast forecast = new Forecast(sample.size() + predictions);

    // Skip

    return forecast;
  }

  /**
   * Sets the weights to use for computing the weighted average for observed
   * values.
   *
   * @param weights The weights to use for computing the weighted average for
   *                observed values.
   */
  private void setWeights(final double[] weights)
  {
    if (weights == null || weights.length < 2)
    {
      // If no weights have been assigned specifically, fall back to the simple
      // average model.
      this.weights = new double[] { 1 };
    }
    else
    {
      // Find the total for the weights.
      double total = 0;
      for (final double weight : weights)
      {
        total += weight;
      }

      // Check if the weights need to be flattened out.
      final boolean flatten = total > 1;

      // Save the weights.
      this.weights = new double[weights.length];
      for (int i = 0; i < weights.length; ++i)
      {
        this.weights[i] = flatten ? weights[i] / total : weights[i];
      }
    }
  }
}
