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
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;

import java.util.Arrays;

/**
 * <p>
 * Generates forecast for a sample that contains upward and/or downward trends,
 * using double exponential smoothing, where an observation \(\bf y_t\) is
 * dampened exponentially to get a level estimate \(\bf l_t\) and a trend
 * estimate \(\bf b_t\) as
 * </p>
 *
 * <p>
 * <br>
 * \(\large
 * \begin{align}
 * l_t &amp;= \alpha{y_t} + (1 - \alpha)(l_{t-1} + b_{t-1}) \tag*{level} \\
 * b_t &amp;= \beta(l_t - l_{t-1}) + (1 - \beta)b_{t-1} \tag*{trend}
 * \end{align}
 * \)
 * </p>
 *
 * <p>
 * where, \(\bf t\) is an index that ranges from {@literal 1} to the number of
 * observations in the sample, \(\bf \alpha\) and \(\bf \beta\) are dampening
 * factors between \(\bf 0.0\) and \(\bf 1.0\) responsible for smoothing out
 * the observations, \(\bf y_t\) is the {@literal t-th} observation,
 * \(\bf l_t\) its level estimate,, and \(\bf b_t\) is an estimate of the
 * upward or downward trend for the observation. This means
 * </p>
 *
 * <p>
 * \(\large l_2 = \alpha{y_2} + (1 - \alpha)(l_1 + b_1)\)
 * <br>
 * \(\large l_3 = \alpha{y_3} + (1 - \alpha)(l_2 + b_2)\)
 * <br>
 * \(\large l_4 = \alpha{y_4} + (1 - \alpha)(l_3 + b_2)\)
 * <br>
 * ... and so on, and
 * </p>
 *
 * <p>
 * \(\large b_2 = \beta(l_2 - l_1) + (1 - \beta)b_1\)
 * <br>
 * \(\large b_3 = \beta(l_3 - l_2) + (1 - \beta)b_2\)
 * <br>
 * \(\large b_4 = \beta(l_4 - l_3) + (1 - \beta)b_3\)
 * <br>
 * ... and so on.
 * </p>
 *
 * <p>
 * The forecast \(f_t\) corresponding to the observation \(y_t\) is calculated
 * as
 * </p>
 *
 * <p>
 * <br>
 * \(\large f_t = l_t + b_t\)
 * <br>
 * </p>
 *
 * <p>
 * and a forecast \(h\) points beyond the available sample is calculated as
 * </p>
 *
 * <p>
 * <br>
 * \(\large f_{t+h} = l_t + h{b_t} \tag*{forecast}\)
 * <br>
 * </p>
 *
 * <p>
 * where, \(l_t\) is the last level estimate, and \(b_t\) is the last trend
 * estimate.
 * </p>
 *
 * <p>
 * The initial values are chosen as
 * </p>
 *
 * <p>
 * <br>
 * \(\large l_1 = y_1\), and
 * <br>
 * \(\large b_1 = \frac{y_n - y_1}{n - 1}\),
 * <br><br>
 * where, \(\bf n\) is the number of observations in the sample, \(\bf y_1\)
 * is the first (chronologically oldest) observation, and \(\bf y_n\) is the
 * last (chronologically latest) observation.
 * </p>
 *
 * <p>
 * Double exponential smoothing improves over single exponential smoothing by
 * directly adding the effect of an upward or downward trend in the sample
 * to the forecast. This enables the forecast to stay closer to the sample
 * and reduces the lag between corresponding observed and predicted values.
 * </p>
 *
 * <p>
 * The optimal values of \(\alpha\) and \(\beta\) are determined through an
 * iterative process for each sample, such that the forecast approximates
 * the sample as closely as possible.
 * </p>
 *
 * @see <a href="https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc433.htm">Double Exponential Smoothing</a>
 */
public class DoubleExponentialSmoothingForecastModel extends ExponentialSmoothingForecastModel
{
  private static final BOBYQAOptimizer OPTIMIZER = new BOBYQAOptimizer(5);

  /**
   * {@inheritDoc}
   */
  @Override
  Forecast forecast(final double[] observations, final int projections)
  {
    // Find optimal values for the dampening factors that will provide a
    // forecast that most closely fits the observations.
    final double[] optimalDampeningFactors = optimalDampeningFactors(observations);
    final double alpha = optimalDampeningFactors[0], beta = optimalDampeningFactors[1];

    // Smoothen the observations using the optimal values for alpha and beta.
    final double[][] smoothed = smoothenObservations(observations, alpha, beta);
    final double[] forecast = smoothed[0], level = smoothed[1], trend = smoothed[2];

    final int samples = observations.length;

    // Add the forecasts for the known observations.
    final double[] predictions = Arrays.copyOf(forecast, samples + projections);

    // Add specified number of predictions beyond the sample.
    for (int i = 0; i < projections; ++i)
    {
      predictions[samples + i] = level[samples - 1] + (i + 1) * trend[samples - 1];
    }

    return forecast(observations, predictions);
  }

  /**
   * <p>
   * Gets an estimate for the overall trend for a collection of observations,
   * which is a measure of whether the data moved upwards or downwards from
   * the first observation to the last.
   * </p>
   *
   * <p>
   * The trend is calculated as
   * </p>
   *
   * <p>
   * <br>
   * \(\large b_1 = \frac{y_n - y_1}{n - 1}\)
   * <br>
   * </p>
   *
   * <p>
   * where, \(n\) is the number of observations, \(y_1\) is the first
   * (chronologically oldest) observation, and \(y_n\) is the last
   * (chronologically latest) observation.
   * </p>
   *
   * @param observations The observations for which the trend is required.
   * @return An estimate for the overall trend for the specified observations.
   */
  private double estimatedInitialTrend(final double[] observations)
  {
    return (observations[observations.length - 1] - observations[0]) / Math.max(1, observations.length - 1);
  }

  /**
   * Gets optimal values for the dampening factors \(\alpha\) and \(\beta\)
   * for a set of observations.
   *
   * @param observations The observations for which \(\alpha\) and \(\beta\)
   *                     are required.
   * @return Best-effort optimal values for \(\alpha\) and \(\beta\), given the
   * observations.
   */
  private double[] optimalDampeningFactors(final double[] observations)
  {
    return OPTIMIZER.optimize(optimizationModelFunction(observations)
        , OPTIMIZATION_GOAL
        , new InitialGuess(new double[] { INITIAL_DAMPENING_FACTOR, INITIAL_DAMPENING_FACTOR })
        , new SimpleBounds(new double[] { MIN_DAMPENING_FACTOR, MIN_DAMPENING_FACTOR }, new double[] { MAX_DAMPENING_FACTOR, MAX_DAMPENING_FACTOR })
        , new MaxEval(MAX_OPTIMIZATION_EVALUATIONS)
        , new MaxIter(MAX_OPTIMIZATION_ITERATIONS)).getPoint();
  }

  /**
   * Calculates the SSE for specified observations and given values of
   * \(\alpha\) and \(\beta\), so that the optimal values of the dampening
   * factors can be determined iteratively.
   *
   * @param observations The observations for which optimal values of
   *                     \(\alpha\) and \(\beta\) are required.
   * @return An {@link ObjectiveFunction}.
   */
  private ObjectiveFunction optimizationModelFunction(final double[] observations)
  {
    return new ObjectiveFunction(params -> {
      final int samples = observations.length;
      final double alpha = params[0], beta = params[1];

      // Smoothen the observations.
      final double[] forecast = smoothenObservations(observations, alpha, beta)[0];

      // Calculate SSE.
      double sse = 0.0;
      for (int i = 0; i < samples - 1; ++i)
      {
        final double error = observations[i] - forecast[i];

        sse += error * error;
      }

      return sse;
    });
  }

  /**
   * Smoothens a collection of observations by exponentially dampening them
   * using factors \(\alpha\) and \(\beta\).
   *
   * @param observations The observations to smoothen.
   * @param alpha        The dampening factor \(\alpha\).
   * @param beta         The dampening factor \(\beta\).
   * @return A three-dimension array where the first dimension contains the
   * forecast, the second dimension contains level estimates for the
   * observations, and the third dimension contains trend estimates.
   */
  private double[][] smoothenObservations(final double[] observations, final double alpha, final double beta)
  {
    // Determine the sample size.
    final int samples = observations.length;

    final double[] forecast = new double[samples], level = new double[samples], trend = new double[samples];

    // Use the first observation as the first smooth observation.
    level[0] = observations[0];

    // Estimate the overall trend using the first and last observations.
    trend[0] = estimatedInitialTrend(observations);

    // Set the value for the first prediction.
    forecast[0] = level[0] + trend[0];

    // Generate the rest using the smoothing formula.
    for (int t = 1; t < samples; ++t)
    {
      level[t] = alpha * observations[t] + (1 - alpha) * (level[t - 1] + trend[t - 1]);

      // Update the trend.
      trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1];

      // Generate the forecast.
      forecast[t] = level[t - 1] + trend[t - 1];
    }

    return new double[][] { forecast, level, trend };
  }
}
