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
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
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
 * using double exponential smoothing, where an observation \(\bf O\) is
 * dampened exponentially to get a smooth version \(\bf S\) as
 * </p>
 *
 * <p>
 * <br>
 * \(\large \boxed{l_t = \alpha{y_t} + (1 - \alpha)(l_{t-1} + b_{t-1})}\), and
 * <br>
 * \(\large \boxed{b_t = \beta(l_t - l_{t-1}) + (1 - \beta)b_{t-1}}\)
 * </p>
 *
 * <p>
 * where, \(\bf t\) is an index that ranges from {@literal 1} to the number of
 * observations in the sample, \(\bf y_t\) is the {@literal t-th} observation,
 * \(\bf l_t\) its smooth version, \(\bf \alpha\) and \(\bf \beta\) are
 * dampening factors between \(\bf 0.0\) and \(\bf 1.0\) responsible for
 * smoothing out the observations, and \(\bf b_t\) is an estimate of the upward
 * or downward trend for the {@literal t-th} observation. This means
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
 * The initial values are chosen as
 * </p>
 *
 * <p>
 * <br>
 * \(\large \boxed{l_1 = y_1}\), and
 * <br>
 * \(\large \boxed{b_1 = \frac{y_n - y_1}{n - 1}}\),
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
    final double[][] processed = smoothenObservations(observations, alpha, beta);
    final double[] smoothed = processed[0], trend = processed[1];

    // Add the smooth observations as the predictions for the known
    // observations.
    final double[] predictions = Arrays.copyOf(smoothed, observations.length + projections);

    // Add specified number of predictions beyond the sample.
    for (int i = 0; i < projections; ++i)
    {
      predictions[observations.length + i] = smoothed[observations.length - 1] + (i + 1) * trend[observations.length - 1];
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
   * Gets the model function to use for optimizing the values of \(\alpha\)
   * and \(\beta\), which is calculates the SSE for the given values of
   * \(\alpha\) and \(\beta\).
   *
   * @param observations The observations for which optimal values of
   *                     \(\alpha\) and \(\beta\) are required.
   * @return A {@link MultivariateVectorFunction}.
   */
  private ObjectiveFunction optimizationModelFunction(final double[] observations)
  {
    return new ObjectiveFunction(params -> {
      // Smoothen the observations.
      final double[] smoothed = smoothenObservations(observations, params[0], params[1])[0];

      // Calculate SSE, starting with the first forecast (corresponding
      // with the second observation in the series).
      double sse = 0.0;
      for (int i = 0; i < observations.length - 1; ++i)
      {
        // The observations need to be compared to forecasts that are one
        // ahead. This is because the forecasts depend directly upon the
        // observations, so, when observations are compared directly to the
        // corresponding forecasts, error will be minimized when alpha = 1.0.
        // This will automatically guide the optimizer to seek the value of
        // 1.0 for alpha, which would make the forecasts useless.
        final double error = observations[i] - smoothed[i + 1];

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
   * @return A two-dimension array where the first dimension contains the
   * smoothened observations, and the second dimension contains an estimate of
   * the trend at each of the observed points.
   */
  private double[][] smoothenObservations(final double[] observations, final double alpha, final double beta)
  {
    // Determine the sample size.
    final int samples = observations.length;

    final double[] level = new double[samples], trend = new double[samples];

    // Use the first observation as the first smooth observation.
    level[0] = observations[0];

    // Estimate the overall trend using the first and last observations.
    trend[0] = estimatedInitialTrend(observations);

    // Generate the rest using the smoothing formula.
    for (int i = 1; i < samples; ++i)
    {
      level[i] = alpha * observations[i] + (1 - alpha) * (level[i - 1] + trend[i - 1]);

      // Update the trend.
      trend[i] = beta * (level[i] - level[i - 1]) + (1 - beta) * trend[i - 1];
    }

    return new double[][] { level, trend };
  }
}
