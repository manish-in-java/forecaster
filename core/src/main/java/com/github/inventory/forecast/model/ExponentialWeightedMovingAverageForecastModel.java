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
import org.apache.commons.math3.analysis.MultivariateMatrixFunction;

import java.util.Arrays;

/**
 * <p>
 * Generates forecast for a sample using exponential smoothing, where the
 * smooth version \(S\) of an observation \(O\) is obtained as
 * </p>
 *
 * <p>
 * <br>
 * \(S_i = \alpha{O_i} + (1 - \alpha)S_{i-1}\)
 * <br>
 * </p>
 *
 * <p>or its alternative form</p>
 *
 * <p>
 * <br>
 * \(S_i = S_{i-1} + \alpha({O_i} - S_{i-1})\)
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
   * {@inheritDoc}
   */
  @Override
  Forecast forecast(final double[] observations, final int projections)
  {
    // Using the initial guess for alpha, find its optimal value that will
    // provide a forecast that most closely fits the observations.
    final double optimalAlpha = optimalAlpha(observations);

    // Smoothen the observations.
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

  /**
   * <p>
   * Gets the {@literal Jacobian} corresponding to the model function used
   * for optimizing the value of \(\alpha\).
   * </p>
   * <p>
   * The {@literal Jacobian} (\(J\)) for a function \(S\) of \(k\) parameters
   * \(x_k\) is a matrix such that an element \(J_{ik}\) of the matrix is
   * given by \(J_{ik} = \frac{\partial S_i}{\partial x_k}\), where \(x_k\)
   * are the \(k\) parameters on which the function \(S\) is dependent, and
   * \(S_i\) are the values of the function \(S\) at \(i\) distinct points. For
   * a function \(S\) that is differentiable with respect to \(x_k\), the
   * {@literal Jacobian} (\(J\)) is a good linear approximation of the
   * geometric shape of the function \(S\) in the immediate vicinity of each
   * \(x_k\). This allows it to be used as a sort of derivative for the
   * function \(S\), wherever a derivative is required to determine the
   * slope of the function at any given point.
   * </p>
   * <p>
   * In the case of single exponential smoothing forecast models, there is
   * only one parameter \(\alpha\), since the predicted values only depend
   * on this one parameter. Therefore, the {@literal Jacobian} (\(J\))
   * depends on only this one parameter \(\alpha\). Therefore, (\(J\))
   * reduces to \(J_{i} = \frac{\partial S_i}{\partial \alpha}\) (\(k = 1\)).
   * </p>
   * <p>
   * For {@literal EWMA}, the function \(S\) is defined as
   * </p>
   * <p>
   * <br>
   * \(S_i = S_{i-1} + \alpha(O_i - S_{i-1})\)
   * <br>
   * </p>
   * <p>
   * therefore, \(J_{i} = \frac{\partial S_i}{\partial \alpha}\) reduces to
   * \(J_{i} = O_i - S_{i-1}\).
   * </p>
   *
   * @param observations The observations for which the optimal value of
   *                     \(\alpha\) is required.
   * @return A {@link MultivariateMatrixFunction}.
   * @see <a href="https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant">Jacobian matrix</a>
   */
  @Override
  MultivariateMatrixFunction optimizationModelJacobian(final double[] observations)
  {
    return params -> {
      // Smoothen the observations.
      final double[] smoothed = smoothenObservations(observations, params[0]);

      final double[][] jacobian = new double[observations.length][1];

      // The first element of the Jacobian is simply the first observation,
      // since there is no prior prediction for it.
      jacobian[0][0] = observations[0];

      // Calculate the rest of the Jacobian using the current observation
      // and the immediately previous prediction.
      for (int i = 1; i < jacobian.length; ++i)
      {
        jacobian[i][0] = observations[i] - smoothed[i - 1];
      }

      return jacobian;
    };
  }

  /**
   * Smoothens a collection of observations by exponentially smoothing them
   * using a dampening factor \(\alpha\), using each observation to generate
   * its corresponding (instead of next) prediction.
   *
   * @param observations The observations to smoothen.
   * @param alpha        The dampening factor \(\alpha\).
   * @return Smoothened observations.
   */
  @Override
  double[] smoothenObservations(final double[] observations, final double alpha)
  {
    final double[] smoothed = new double[observations.length];

    // Generate the first smooth observation using a specific strategy.
    smoothed[0] = firstPrediction(observations);

    // Generate the rest using the smoothing formula.
    for (int i = 1; i < observations.length; ++i)
    {
      smoothed[i] = alpha * observations[i] + (1 - alpha) * smoothed[i - 1];
    }

    return smoothed;
  }
}
