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

import java.util.Arrays;

/**
 * <p>
 * Generates forecast for a sample using exponential smoothing, where the
 * smooth version \(S\) of an observation \(O\) is obtained as
 * </p>
 *
 * <p>
 * <br>
 * \(\boxed{S_i = \alpha{O_i} + (1 - \alpha)S_{i-1}}\)
 * <br>
 * </p>
 *
 * <p>or its alternative form</p>
 *
 * <p>
 * <br>
 * \(\boxed{S_i = S_{i-1} + \alpha({O_i} - S_{i-1})}\)
 * <br>
 * </p>
 *
 * <p>
 * where, \(i\) is an index that ranges from {@literal 1} to the number of
 * observations in the sample, \(O_i\) is the {@literal i-th} observation,
 * \(S_i\) its smooth version, and \(\alpha\) is a dampening factor between
 * \(0\) and \(1\) responsible for smoothing out the observations. This means
 * </p>
 *
 * <p>
 * \(S_2 = \alpha{O_2} + (1 - \alpha)S_1\)
 * <br>
 * \(S_3 = \alpha{O_3} + (1 - \alpha)S_2\)
 * <br>
 * \(S_4 = \alpha{O_4} + (1 - \alpha)S_3\)
 * <br><br>
 * and so on.
 * </p>
 *
 * <p>
 * \(S_1\) can be chosen using one of many possible strategies, one of which
 * must be provided at the time of initializing the model.
 * </p>
 *
 * <p>
 * This variation, also known by its short form {@literal EWMA} is
 * characterized by a direct dependence of \(S_i\) upon \(O_i\), and was
 * originally proposed in {@literal 1959} by {@literal S.W. Roberts}.
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
 * The initial value of \(\alpha\) must be between \(0.0\) and \(1.0\). The
 * closer the value is to \(0.0\), lesser the contribution of observations
 * farther in the past and higher that of most recent predictions. The closer
 * it is to \(1.0\), lesser the contribution of predictions and higher that
 * of the observations.
 * </p>
 *
 * @see <a href="https://en.wikipedia.org/wiki/EWMA_chart">Exponential Weighted Moving Average</a>
 */
public class ExponentialWeightedMovingAverageForecastModel extends SingleExponentialSmoothingForecastModel
{
  /**
   * Creates a model with simple average strategy for generating the first
   * prediction and a {@literal non-linear gradient-descent} optimizer for
   * finding the optimal value for the dampening factor \(\alpha\).
   */
  public ExponentialWeightedMovingAverageForecastModel()
  {
    super(AlphaOptimizer.GRADIENT_DESCENT);
  }

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
