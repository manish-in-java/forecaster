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
 * Generates forecast for a sample that shows seasonality (periodicity) in
 * addition to upward and/or downward trends, using triple exponential
 * smoothing, where an observation \(\bf y_t\) is dampened exponentially to
 * get a level estimate \(\bf l_t\), a trend estimate \(\bf b_t\) and a
 * seasonality estimate \(\bf s_t\) as
 * </p>
 *
 * <p>
 * <br>
 * \(\large
 * \begin{align}
 * l_t &amp;= \alpha(y_t - s_{t-m}) + (1 - \alpha)(l_{t-1} + b_{t-1}) \tag*{level} \\
 * b_t &amp;= \beta(l_t - l_{t-1}) + (1 - \beta)b_{t-1} \tag*{trend} \\
 * s_t &amp;= \gamma(y_t - l_t) + (1 - \gamma)s_{t-m} \tag*{seasonality}
 * \end{align}
 * \)
 * </p>
 *
 * <p>
 * where, \(\bf t\) is an index that ranges from {@literal 1} to the number of
 * observations in the sample, \(\bf \alpha\), \(\bf \beta\) and \(\bf \gamma\)
 * are dampening factors between \(\bf 0.0\) and \(\bf 1.0\) responsible for
 * smoothing out the observations, \(\bf m\) is the number of observations in a
 * single season, \(\bf y_t\) is the {@literal t-th} observation, \(\bf l_t\)
 * its level estimate, \(\bf b_t\) an estimate of the upward or downward trend
 * for \(y_t\), and \(\bf s_t\) an estimate of seasonality, known as a
 * seasonality index.
 * </p>
 *
 * <p>
 * A forecast \(k\) points beyond the available sample is calculated as
 * </p>
 *
 * <p>
 * <br>
 * \(\large f_{t+k} = l_t + k{b_t} + s_{t-m+k} \tag*{forecast}\)
 * <br>
 * </p>
 *
 * <p>
 * where, \(l_t\) is the last level estimate, \(b_t\) the last trend estimate,
 * and \(s_{t-m}\) is the seasonality estimate for the same period during the
 * last season.
 * </p>
 *
 * <p>
 * This version is known as <i>additive triple exponential smoothing</i> due to
 * the fact that the effect of seasonality is <i>added</i> to the observed
 * value (\(l_{m+1} = \alpha(y_{m+1} - s_1) + ...\)) when smoothing the
 * observations. Another version, known as the <i>multiplicative triple
 * exponential smoothing</i> is expressed as
 * </p>
 *
 * <p>
 * <br>
 * \(\large
 * \begin{align}
 * l_t &amp;= \alpha\frac{y_t}{s_{t-m}} + (1 - \alpha)(l_{t-1} + b_{t-1}) \tag*{level} \\
 * b_t &amp;= \beta(l_t - l_{t-1}) + (1 - \beta)b_{t-1} \tag*{trend} \\
 * s_t &amp;= \gamma\frac{(y_t}{l_t} + (1 - \gamma)s_{t-m} \tag*{seasonality}
 * \end{align}
 * \)
 * </p>
 *
 * <p>
 * A forecast \(k\) points beyond the available sample is calculated as
 * </p>
 *
 * <p>
 * <br>
 * \(\large f_{t+k} = (l_t + k{b_t})s_{t-m+k} \tag*{forecast}\)
 * <br>
 * </p>
 *
 * <p>
 * In the context of forecasting, the term <i>seasonality</i> simply refers to
 * some sort of periodicity in the trend, and not necessarily to seasons of the
 * year. This is another way of saying that the sample has some sort of a
 * repetitive pattern for the trend. For example, if a sample shows a trend
 * like <i>up-up-down-up-up-down-up-up-down-up-up-down</i>, it is said to have
 * a seasonality of <i>up-up-down</i> in the trend, with a period of \(\bf 3\)
 * (the number of observations in a single <i>up-up-down</i> season).
 * </p>
 *
 * <p>
 * Given that \(l_t\) depends on \(s_{t-m}\), the smooth values cannot be
 * calculated for observations up to \(y_m\) (since \(i-m\) is negative and the
 * corresponding \(s_{t-m}\) are unavailable). Therefore, the smooth value
 * \(l_m\) is chosen upfront as
 * </p>
 *
 * <p>
 * <br>
 * \(\large l_m = y_1\)
 * </p>
 *
 * <p>
 * The initial estimate for the trend, \(b_m\) is calculated as
 * </p>
 *
 * <p>
 * <br>
 * \(\large b_m = \frac{1}{m^2}[(y_{m+1} - y_1) + (y_{m+2} - y_2) + ... + (y_{2m} - y_m)]\)
 * </p>
 *
 * <p>
 * Given the dependence of the seasonality index \(s_t\) on \(s_{t-m}\), the
 * first \(m\) seasonality indices \(s_1\) to \(s_m\) also need to be
 * estimated. For the model to produce an optimum forecast, this requires
 * a slightly involved procedure, as explained below.
 * </p>
 *
 * <p>
 * <b>Step 1:</b> Calculate the number of seasons \(n\) for the observations,
 * using the number of observations in a single season. For example, if
 * \(m = 4\) and there are 12 observations in all, the number of seasons will
 * be \(n = 12 / 4 = 3\).
 * </p>
 *
 * <p>
 * <b>Step 2:</b> Calculate the simple average for each season as
 * </p>
 *
 * <p>
 * <br>
 * \(\large a_1 = \frac{1}{m}(y_1 + y_2 + ... + y_m)\)
 * <br>
 * \(\large a_2 = \frac{1}{m}(y_{m+1} + y_{m+2} + ... + y_{2m})\)
 * <br>
 * ...
 * <br>
 * \(\large a_n = \frac{1}{m}(y_{(n-1)m+1} + y_{(n-1)m+2} + ... + y_{nm})\)
 * </p>
 *
 * <p>
 * <b>Step 3:</b> Calculate the first \(m\) seasonality indices \(s_1\) to
 * \(s_m\) as
 * </p>
 *
 * <p>
 * <br>
 * \(\large s_1 = \frac{1}{n}(\frac{y_1}{a_1} + \frac{y_{m+1}}{a_2} + ... + \frac{y_{(n-1)m+1}}{a_n})\)
 * <br>
 * \(\large s_2 = \frac{1}{n}(\frac{y_2}{a_1} + \frac{y_{m+2}}{a_2} + ... + \frac{y_{(n-1)m+2}}{a_n})\)
 * <br>
 * ...
 * <br>
 * \(\large s_m = \frac{1}{n}(\frac{y_m}{a_1} + \frac{y_{2m}}{a_2} + ... + \frac{y_{nm}}{a_n})\)
 * </p>
 *
 * <p>
 * The optimal values of \(\alpha\), \(\beta\) and \(\gamma\) are determined
 * through an iterative process for each sample, such that the forecast
 * approximates the sample as closely as possible.
 * </p>
 *
 * <p>
 * The forecast \(f_t\) corresponding to the observation \(y_t\) is calculated
 * as
 * </p>
 *
 * <p>
 * This model is also known as <i>Holt-Winters model</i> after its inventors
 * <i>C.C. Holt</i> and <i>Peter Winters</i> who proposed it in <i>1965</i>.
 * </p>
 *
 * @see <a href="https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm">Triple Exponential Smoothing</a>
 */
public class TripleExponentialSmoothingForecastModel extends ExponentialSmoothingForecastModel
{
  private static final BOBYQAOptimizer OPTIMIZER = new BOBYQAOptimizer(7);

  private final boolean multiplicative;
  private final int     periodsPerSeason;

  /**
   * Creates a model with a specified number of periods per season in the
   * sample.
   *
   * @param periodsPerSeason The number of periods per season for the model.
   * @param multiplicative   Whether the seasonality is multiplicative.
   * @throws IllegalArgumentException if {@code periodsPerSeason} is
   *                                  {@literal zero} or negative.
   */
  public TripleExponentialSmoothingForecastModel(final int periodsPerSeason, final boolean multiplicative)
  {
    if (periodsPerSeason < 1)
    {
      throw new IllegalArgumentException("Periods per season must be greater than zero.");
    }

    this.multiplicative = multiplicative;
    this.periodsPerSeason = periodsPerSeason;
  }

  /**
   * {@inheritDoc}
   *
   * @throws IllegalArgumentException if the number of observations is not a
   *                                  whole-number multiple of the number of
   *                                  periods per season, or if the number of
   *                                  observations is less than the number of
   *                                  periods for two seasons, which is the
   *                                  minimum number of periods required for
   *                                  the model to work.
   */
  @Override
  Forecast forecast(final double[] observations, final int projections)
  {
    final int samples = observations.length;

    // Ensure that the number of observations is a multiple of periods per
    // season as the model cannot work otherwise.
    if (samples % periodsPerSeason != 0)
    {
      throw new IllegalArgumentException("The number of observations ("
                                             + samples + ") must be a whole-number multiple of the number of periods per season ("
                                             + periodsPerSeason + ").");
    }

    // Ensure that the number of observations is at least two times the
    // number of periods.
    if (samples < 2 * periodsPerSeason)
    {
      throw new IllegalArgumentException("The number of observations ("
                                             + samples + ") must be at least two times the number of periods per season ("
                                             + periodsPerSeason + ").");
    }

    // Find optimal values for the dampening factors that will provide a
    // forecast that most closely fits the observations.
    final double[] optimalDampeningFactors = optimalDampeningFactors(observations);
    final double alpha = optimalDampeningFactors[0], beta = optimalDampeningFactors[1], gamma = optimalDampeningFactors[2];

    System.out.println("Alpha = " + alpha + ", Beta = " + beta + ", Gamma = " + gamma);

    // Smoothen the observations using the optimal values for alpha, beta and gamma.
    final double[][] processed = smoothenObservations(observations, alpha, beta, gamma);
    final double[] forecast = processed[0], level = processed[1], seasonality = processed[3], trend = processed[2];

    for (int t = 0; t < samples; ++t)
    {
      System.out.println(String.format("%4.6f    %4.6f    %4.6f    %4.6f    %4.6f", observations[t], level[t], trend[t], seasonality[t], forecast[t]));
    }

    // Add the smooth observations as the predictions for the known
    // observations.
    final double[] predictions = Arrays.copyOf(forecast, samples + projections);

    // Add specified number of predictions beyond the sample.
    for (int i = 0; i < projections; ++i)
    {
      // For predictions beyond the available sample, the baseline period
      // should be within the last season. Determine the baseline period
      // for the current prediction.
      final int period = samples - periodsPerSeason + i % periodsPerSeason - 1;

      predictions[samples + i] = level[samples - 1] + (i + 1) * trend[samples - 1] + seasonality[period];
    }

    return forecast(observations, predictions);
  }

  /**
   * <p>
   * Gets an estimate for the initial trend for a collection of observations,
   * as
   * </p>
   *
   * <p>
   * <br>
   * \(\large b_1 = \frac{1}{m^2}[(y_{m+1} - y_1) + (y_{m+2} - y_2) + ... + (y_{2m} - y_m)]\)
   * <br>
   * </p>
   *
   * <p>
   * where, \(m\) is the number of observations in a single season and
   * \(y_1\) is the first (chronologically oldest) observation.
   * </p>
   *
   * @param observations The observations for which the trend is required.
   * @return An estimate for the initial trend for the specified observations.
   */
  private double estimatedInitialTrend(final double[] observations)
  {
    // Calculate the estimate for the initial trend.
    double delta = 0.0;
    for (int i = 0; i < periodsPerSeason; ++i)
    {
      delta += observations[periodsPerSeason + i] - observations[i];
    }

    return delta / (periodsPerSeason * periodsPerSeason);
  }

  /**
   * Gets optimal values for the dampening factors \(\alpha\), \(\beta\) and
   * \(\gamma\) for a set of observations.
   *
   * @param observations The observations for which \(\alpha\), \(\beta\) and
   *                     \(\gamma\) are required.
   * @return Best-effort optimal values for \(\alpha\), \(\beta\) and
   * \(\gamma\), given the observations.
   */
  private double[] optimalDampeningFactors(final double[] observations)
  {
    return OPTIMIZER.optimize(optimizationModelFunction(observations)
        , OPTIMIZATION_GOAL
        , new InitialGuess(new double[] { INITIAL_DAMPENING_FACTOR, INITIAL_DAMPENING_FACTOR, INITIAL_DAMPENING_FACTOR })
        , new SimpleBounds(new double[] { MIN_DAMPENING_FACTOR, MIN_DAMPENING_FACTOR, MIN_DAMPENING_FACTOR },
                           new double[] { MAX_DAMPENING_FACTOR, MAX_DAMPENING_FACTOR, MAX_DAMPENING_FACTOR })
        , new MaxEval(MAX_OPTIMIZATION_EVALUATIONS)
        , new MaxIter(MAX_OPTIMIZATION_ITERATIONS)).getPoint();
  }

  /**
   * Calculates the SSE for specified observations and given values of
   * \(\alpha\), \(\beta\) and \(\gamma\), so that the optimal values of the
   * dampening factors can be determined iteratively.
   *
   * @param observations The observations for which optimal values of
   *                     \(\alpha\), \(\beta\) and \(\gamma\) are required.
   * @return An {@link ObjectiveFunction}.
   */
  private ObjectiveFunction optimizationModelFunction(final double[] observations)
  {
    return new ObjectiveFunction(params -> {
      final int samples = observations.length;
      final double alpha = params[0], beta = params[1], gamma = params[2];

      // Smoothen the observations.
      final double[] forecast = smoothenObservations(observations, alpha, beta, gamma)[0];

      // Calculate SSE.
      double sse = 0.0;
      for (int t = periodsPerSeason; t < samples; ++t)
      {
        final double error = observations[t] - forecast[t];

        sse += error * error;
      }

      return sse;
    });
  }

  /**
   * Gets the season-wise simple averages for observations.
   *
   * @param observations The observations to average.
   * @return The season-wise simple averages for observations.
   */
  private double[] seasonalAverages(final double[] observations)
  {
    final int seasons = observations.length / periodsPerSeason;

    final double[] averages = new double[seasons];

    for (int i = 0; i < seasons; ++i)
    {
      for (int j = 0; j < periodsPerSeason; ++j)
      {
        averages[i] += observations[i * periodsPerSeason + j];
      }

      averages[i] /= periodsPerSeason;
    }

    return averages;
  }

  /**
   * Smoothens a collection of observations by exponentially dampening them
   * using factors \(\alpha\), \(\beta\) and \(\gamma\).
   *
   * @param observations The observations to smoothen.
   * @param alpha        The dampening factor \(\alpha\).
   * @param beta         The dampening factor \(\beta\).
   * @param gamma        The dampening factor \(\gamma\).
   * @return A four-dimension array where the first dimension contains the
   * forecast corresponding to the observations, the second dimension contains
   * the level estimates, the third dimension the estimates, and the fourth
   * the seasonality estimates.
   */
  private double[][] smoothenObservations(final double[] observations, final double alpha, final double beta, final double gamma)
  {
    // Determine the sample size and the number of seasons for the sample.
    final int samples = observations.length, seasons = samples / periodsPerSeason;

    // Determine seasonal averages, as they are required to calculate the
    // initiate estimates for seasonality.
    final double[] seasonalAverages = seasonalAverages(observations);

    final double[] forecast = new double[samples], level = new double[samples], seasonality = new double[samples], trend = new double[samples];

    for (int t = 0; t < samples; ++t)
    {
      if (t == periodsPerSeason - 1)
      {
        // Use the first observation as the initial estimate for the level.
        level[t] = observations[0];

        // Determine an initial estimate for the trend.
        trend[t] = estimatedInitialTrend(observations);
      }
      else if (t >= periodsPerSeason)
      {
        // Calculate the level.
        level[t] = alpha * (multiplicative ? observations[t] / seasonality[t - periodsPerSeason] : observations[t] - seasonality[t - periodsPerSeason])
            + (1 - alpha) * (level[t - 1] + trend[t - 1]);

        // Calculate the trend.
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1];

        // Calculate the forecast.
        forecast[t] = multiplicative
                      ? (level[t - 1] + trend[t - 1]) * seasonality[t - periodsPerSeason]
                      : level[t - 1] + trend[t - 1] + seasonality[t - periodsPerSeason];
      }

      if (t < periodsPerSeason)
      {
        // For the first season, estimate the seasonality indices.
        for (int i = 0; i < seasons; ++i)
        {
          seasonality[t] += observations[t + i * periodsPerSeason] / seasonalAverages[i];
        }

        seasonality[t] /= seasons;
      }
      else
      {
        // Calculate the seasonality index.
        seasonality[t] = gamma * (multiplicative ? observations[t] / level[t] : observations[t] - level[t]) + (1 - gamma) * seasonality[t - periodsPerSeason];
      }
    }

    return new double[][] { forecast, level, trend, seasonality };
  }
}
