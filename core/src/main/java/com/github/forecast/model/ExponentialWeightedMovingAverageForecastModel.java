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

import java.util.Arrays;

/**
 * <p>
 * Generates forecast for a sample using exponential smoothing, where an
 * observation \(\bf y_t\) is dampened exponentially to get a smooth version
 * \(\bf l_t\) as
 * </p>
 *
 * <p>
 * <br>
 * \(\large l_t = \alpha{y_t} + (1 - \alpha)l_{t-1}\)
 * <br>
 * </p>
 *
 * <p>or its alternative form</p>
 *
 * <p>
 * <br>
 * \(\large l_t = l_{t-1} + \alpha({y_t} - l_{t-1})\)
 * <br>
 * </p>
 *
 * <p>
 * where, \(t\) is an index that ranges from {@literal 1} to the number of
 * observations in the sample, \(y_t\) is the {@literal t-th} observation,
 * \(l_t\) its smooth version, and \(\alpha\) is a dampening factor between
 * \(0\) and \(1\) responsible for smoothing out the observations. This means
 * </p>
 *
 * <p>
 * \(\large l_2 = \alpha{y_2} + (1 - \alpha)l_1\)
 * <br>
 * \(\large l_3 = \alpha{y_3} + (1 - \alpha)l_2\)
 * <br>
 * \(\large l_4 = \alpha{y_4} + (1 - \alpha)l_3\)
 * <br><br>
 * and so on.
 * </p>
 *
 * <p>
 * \(l_1\) can be chosen using one of many possible strategies, one of which
 * must be provided at the time of initializing the model.
 * </p>
 *
 * <p>
 * The forecast \(f_t\) corresponding to the observation \(y_t\) is the same
 * as \(\l_t\), that is
 * </p>
 *
 * <p>
 * <br>
 * \(\large f_t = l_t\)
 * <br>
 * </p>
 *
 * <p>
 * Smoothing utilizes a dampening factor \(\alpha\), that continually decreases
 * the effect of observations farther in the past. The model starts with an
 * initial value of \(\alpha\) (which is usually guessed) but determines
 * the optimal value for the factor that approximates the sample as closely as
 * possible, through an iterative process. The iterative process attempts to
 * find the optimal value of \(\alpha\) by minimizing the value of the
 * {@literal sum-squared-errors (SSE)} (also referred to as the
 * {@literal cost function} for the model).
 * </p>
 *
 * <p>
 * The value of \(\alpha\) must be between \(0.0\) and \(1.0\). The closer the
 * value is to \(0.0\), lesser the contribution of observations farther in the
 * past and higher that of most recent predictions. The closer it is to
 * \(1.0\), lesser the contribution of predictions and higher that of the
 * observations.
 * </p>
 *
 * <p>
 * This variation, also known by its short form <i>EWMA</i> is characterized
 * by a direct dependence of \(l_t\) upon \(y_t\), and was originally proposed
 * in <i>1959</i> by <i>S.W. Roberts</i>.
 * </p>
 *
 * @see <a href="https://en.wikipedia.org/wiki/EWMA_chart">Exponential Weighted Moving Average</a>
 */
public class ExponentialWeightedMovingAverageForecastModel extends SingleExponentialSmoothingForecastModel
{
  /**
   * Creates a baseline of a collection of observations. The baseline is
   * used to generate predictions for the observations, as well as fine-tune
   * the model to produce optimal predictions.
   *
   * @param observations A collection of observations.
   * @return A baseline version of the observations.
   */
  @Override
  double[] baselineObservations(final double[] observations)
  {
    // In Roberts' model, each observation is its own baseline, that is to say,
    // the prediction for each observation is directly dependent upon that
    // observation itself.
    return observations;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  Forecast forecast(final double[] observations, final int projections)
  {
    // Using the initial guess for alpha, find its optimal value that will
    // provide a forecast that most closely fits the observations.
    final double optimalAlpha = optimalAlpha(observations);

    // Smoothen the observations using the optimal value for alpha.
    final double[] smoothed = smoothenObservations(observations, optimalAlpha);

    // Add the smooth observations as the predictions for the known
    // observations.
    final double[] predictions = Arrays.copyOf(smoothed, observations.length + projections);

    // Add specified number of predictions beyond the sample.
    for (int j = 0; j < projections; ++j)
    {
      // Add the very last prediction generated from the observations to
      // the forecast.
      predictions[observations.length + j] = predictions[observations.length - 1];
    }

    return forecast(observations, predictions);
  }
}
