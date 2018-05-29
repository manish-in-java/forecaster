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
import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.fitting.leastsquares.ParameterValidator;
import org.apache.commons.math3.linear.ArrayRealVector;

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
 * \(\large \boxed{S_i = \alpha{O_i} + (1 - \alpha)(S_{i-1} + T_{i-1})}\), and
 * <br>
 * \(\large \boxed{T_i = \beta(S_i - S_{i-1}) + (1 - \beta)T_{i-1}}\)
 * </p>
 *
 * <p>
 * where, \(\bf i\) is an index that ranges from {@literal 1} to the number of
 * observations in the sample, \(\bf O_i\) is the {@literal i-th} observation,
 * \(\bf S_i\) its smooth version, \(\bf \alpha\) and \(\bf \beta\) are
 * dampening factors between \(\bf 0.0\) and \(\bf 1.0\) responsible for
 * smoothing out the observations, and \(\bf T_i\) is an estimate of the upward
 * or downward trend for the {@literal i-th} observation. This means
 * </p>
 *
 * <p>
 * \(\large S_2 = \alpha{O_2} + (1 - \alpha)(S_1 + T_1)\)
 * <br>
 * \(\large S_3 = \alpha{O_3} + (1 - \alpha)(S_2 + T_2)\)
 * <br>
 * \(\large S_4 = \alpha{O_4} + (1 - \alpha)(S_3 + T_2)\)
 * <br>
 * ... and so on, and
 * </p>
 *
 * <p>
 * \(\large T_2 = \beta(S_2 - S_1) + (1 - \beta)T_1\)
 * <br>
 * \(\large T_3 = \beta(S_3 - S_2) + (1 - \beta)T_2\)
 * <br>
 * \(\large T_4 = \beta(S_4 - S_3) + (1 - \beta)T_3\)
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
 * \(\large \boxed{S_1 = O_1}\), and
 * <br>
 * \(\large \boxed{T_1 = \frac{O_n - O_1}{n - 1}}\),
 * <br><br>
 * where, \(\bf n\) is the number of observations in the sample, \(\bf O_1\)
 * is the first (chronologically oldest) observation, and \(\bf O_n\) is the
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
  /**
   * {@inheritDoc}
   */
  @Override
  Forecast forecast(final double[] observations, final int projections)
  {
    // Find optimal values for the dampening factors that will provide a
    // forecast that most closely fits the observations.
    final double[] optimalDampeningFactors = optimalDampeningFactors(observations);

    // Smoothen the observations using the optimal values for alpha and beta.
    final double[][] smoothed = smoothenObservations(observations, optimalDampeningFactors[0], optimalDampeningFactors[1]);

    // Add the smooth observations as the predictions for the known
    // observations.
    final double[] predictions = Arrays.copyOf(smoothed[0], observations.length + projections);

    // Add specified number of predictions beyond the sample.
    for (int i = 0; i < projections; ++i)
    {
      predictions[observations.length + i] = smoothed[0][observations.length - 1] + (i + 1) * smoothed[1][observations.length - 1];
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
   * \(\large T_1 = \frac{O_n - O_1}{n - 1}\)
   * <br>
   * </p>
   *
   * <p>
   * where, \(n\) is the number of observations, \(O_1\) is the first
   * (chronologically oldest) observation, and \(O_n\) is the last
   * (chronologically latest) observation.
   * </p>
   *
   * @param observations The observations for which the trend is required.
   * @return An estimate for the overall trend for the specified observations.
   */
  private double estimatedTrend(final double[] observations)
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
    return LEAST_SQUARES_OPTIMIZER.optimize(optimizationModel(observations))
                                  .getPoint()
                                  .toArray();
  }

  /**
   * <p>
   * Converts a collection of observations into a
   * {@literal least-squares curve-fitting problem} using an initial
   * guess for the dampening factors \(\alpha\) and \(\beta\).
   * </p>
   *
   * <p>
   * The optimization algorithm requires the following input:
   * </p>
   *
   * <ol>
   * <li>The observations for which \(\alpha\) and \(\beta\) need to be
   * optimized;</li>
   * <li>An initial guess for \(\alpha\) and \(\beta\);</li>
   * <li>A function that can take the observations and some values for
   * \(\alpha\) and \(\beta\), and produce predictions for each of the
   * observations with those values of the dampening factors. This is
   * known as the model function for the algorithm. The algorithm internally
   * calculates the MSE for each value of \(\alpha\) by using the given
   * observations and their corresponding predictions for the specified
   * values of \(\alpha\) and \(beta\);</li>
   * <li>A function that can take the observations and some values for
   * \(\alpha\) and \(\beta\), and produce a <i>Jacobian</i> for each
   * prediction made by the model function. The <i>Jacobian</i> is a linear
   * approximation of the derivative of a predicted value with respect
   * to the variable parameter \(\alpha\) and hence serves to determine
   * whether the trend at each predicted value matches that of the
   * corresponding observed value. This information is critical in optimizing
   * the values of \(\alpha\) and \(beta\) as the derivative determines
   * whether the values are too high or too low, and therefore whether
   * they need to be lowered or raised;</li>
   * <li>A function that can validate whether a specific value of
   * \(\alpha\) or \(\beta\) is within the bounds of the problem-space. For
   * the purposes of exponential smoothing, \(\alpha\) and \(\beta\) must be
   * between {@literal 0.1} and {@literal 0.9}.</li>
   * </ol>
   *
   * @param observations The observations to convert to a least-squares
   *                     curve-fitting problem.
   * @return A {@link LeastSquaresProblem}.
   */
  private LeastSquaresProblem optimizationModel(final double[] observations)
  {
    return new LeastSquaresBuilder()
        .checkerPair(DAMPENING_FACTOR_CONVERGENCE_CHECKER)
        .maxEvaluations(MAX_OPTIMIZATION_EVALUATIONS)
        .maxIterations(MAX_OPTIMIZATION_ITERATIONS)
        .model(optimizationModelFunction(observations), optimizationModelJacobian(observations))
        .parameterValidator(optimizationParameterValidator())
        .target(observations)
        .start(new double[] { INITIAL_DAMPENING_FACTOR, INITIAL_DAMPENING_FACTOR })
        .build();
  }

  /**
   * Gets the model function to use for optimizing the values of \(\alpha\)
   * and \(\beta\), which is just the predicted value for each observed
   * value. The optimization algorithm then uses the difference between the
   * observed and corresponding predicted values to find the optimal values
   * for \(\alpha\) and \(\beta\).
   *
   * @param observations The observations for which optimal values of
   *                     \(\alpha\) and \(\beta\) are required.
   * @return A {@link MultivariateVectorFunction}.
   */
  private MultivariateVectorFunction optimizationModelFunction(final double[] observations)
  {
    return params -> smoothenObservations(observations, params[0], params[1])[0];
  }

  /**
   * <p>
   * Gets the <i>Jacobian</i> (\(J\)) corresponding to the model function used
   * for optimizing the values of \(\alpha\) and \(\beta\). The <i>Jacobian</i>
   * for a function \(S\) of \(k\) parameters \(x_k\) is a matrix, where an
   * element \(J_{ik}\) of the matrix is given by
   * \(J_{ik} = \frac{\partial S_i}{\partial x_k}\), \(x_k\) are the
   * \(k\) parameters on which the function \(S\) is dependent, and \(S_i\) are
   * the values of the function \(S\) at \(i\) distinct points. In the case of
   * the double exponential smoothing forecast model, there are two parameters
   * - \(\alpha\) and \(beta\) that impact the predicted value for a given
   * observed value. Therefore, (\(J\)) contains two values for every \(S_i\)
   * </p>
   *
   * <p>
   * <br>
   * \(\large J_{i\alpha}\), defined as \(\boxed{J_{i\alpha} = \frac{\partial S_i}{\partial \alpha}}\), and
   * <br>
   * \(\large J_{i\beta}\), defined as \(\boxed{J_{i\beta} = \frac{\partial S_i}{\partial \beta}}\).
   * <br>
   * </p>
   *
   * <p>
   * For the double exponential smoothing model, the function \(S\) is
   * defined as (see above)
   * </p>
   *
   * <p>
   * <br>
   * \(\large S_i = \alpha{O_i} + (1 - \alpha)(S_{i-1} + T_{i-1})\)
   * <br>
   * </p>
   *
   * <p>
   * Therefore,
   * </p>
   *
   * <p>
   * <br>
   * \(\large J_{i\alpha} = \frac{\partial S_i}{\partial \alpha}\)
   * <br>
   * becomes (through the rules for partial derivatives and the chain rule of
   * differentiation)
   * <br><br>
   * \(\large J_{i\alpha} = \alpha{\frac{\partial O_i}{\partial \alpha}} + O_i{\frac{\partial \alpha}{\partial \alpha}}
   * + (1 - \alpha)(\frac{\partial S{i-1}}{\partial \alpha} + \frac{\partial T{i-1}}{\partial \alpha})
   * - (S_{i-1} + T_{i-1}){\frac{\partial \alpha}{\partial \alpha}}\),
   * <br>
   * or (since \(\frac{\partial O_i}{\partial \alpha} = 0\), given that \(O_i\)
   * does not depend on \(\alpha\))
   * <br><br>
   * \(\large \boxed{J_{i\alpha} = O_i - S_{i-1} - T_{i-1} + (1 - \alpha)(\frac{\partial S_{i-1}}{\partial \alpha} + \frac{\partial T_{i-1}}{\partial \alpha})}\)
   * </p>
   *
   * <p>
   * Similarly,
   * </p>
   *
   * <p>
   * <br>
   * \(\large J_{i\beta} = \frac{\partial S_i}{\partial \beta}\)
   * <br>
   * becomes (since \(\frac{\partial O_i}{\partial \beta} = \frac{\partial \alpha}{\partial \beta} = 0\),
   * given that \(O_i\) and \(\alpha\) do not depend on \(\beta\))
   * <br><br>
   * \(\large \boxed{J_{i\beta} = (1 - \alpha)(\frac{\partial S_{i-1}}{\partial \beta} + \frac{\partial T_{i-1}}{\partial \beta})}\)
   * </p>
   *
   * <p>
   * Given that \(J_{i\alpha}\) and \(J_{i\beta}\) depend on
   * \(\frac{\partial T_{i-1}}{\partial \alpha}\) and
   * \(\frac{\partial T_{i-1}}{\partial \beta}\) as well
   * </p>
   *
   * <p>
   * <br>
   * \(\large \boxed{\frac{\partial T_i}{\partial \alpha} = \beta(\frac{\partial S_i}{\partial \alpha} - \frac{\partial S_{i-1}}{\partial \alpha})
   * + (1 - \beta)\frac{\partial T_{i-1}}{\partial \alpha}}\), and
   * </p>
   *
   * <p>
   * <br>
   * \(\large \frac{\partial T_i}{\partial \beta} = \beta(\frac{\partial S_i}{\partial \beta} - \frac{\partial S_{i-1}}{\partial \beta})
   * + (S_i - S_{i-1})\frac{\partial \beta}{\partial \beta}
   * + (1 - \beta)\frac{\partial T_{i-1}}{\partial \beta} - T_{i-1}\frac{\partial \beta}{\partial \beta}\), or
   * <br>
   * \(\large \boxed{\frac{\partial T_i}{\partial \beta} = S_i - S_{i-1} - T_{i-1}
   * + \beta(\frac{\partial S_i}{\partial \beta} - \frac{\partial S_{i-1}}{\partial \beta})
   * + (1 - \beta)\frac{\partial T_{i-1}}{\partial \beta}
   * }\)
   * </p>
   *
   * <p>
   * Due to the choice for \(S_1\) and \(T_1\), neither of which depends on
   * \(\alpha\) or \(\beta\),
   * </p>
   *
   * <p>
   * <br>
   * \(\large \boxed{\frac{\partial S_1}{\partial \alpha} =
   * \frac{\partial S_1}{\partial \beta} =
   * \frac{\partial T_1}{\partial \alpha} =
   * \frac{\partial T_1}{\partial \beta} =
   * 0}\)
   * </p>
   *
   * @param observations The observations \(O_i\) for which optimal values of
   *                     \(\alpha\) and \(\beta\) are required.
   * @return A {@link MultivariateMatrixFunction}, which is a two-column
   * matrix whose first column contains elements corresponding to the
   * <i>Jacobian</i> for the model function with respect to the parameter
   * \(\alpha\), and the second column contains elements corresponding to the
   * <i>Jacobian</i> for the model function with respect to the parameter
   * \(\beta\).
   * @see <a href="https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm">Levenberg-Marquardt algorithm</a>
   * @see <a href="https://en.wikipedia.org/wiki/Least_squares">Least squares</a>
   * @see <a href="https://en.wikipedia.org/wiki/Curve_fitting">Curve fitting</a>
   * @see <a href="https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant">Jacobian matrix</a>
   */
  private MultivariateMatrixFunction optimizationModelJacobian(final double[] observations)
  {
    return params -> {
      final double alpha = params[0], beta = params[1];

      // Smoothen the observations.
      final double[] smoothed = smoothenObservations(observations, alpha, beta)[0];

      // Prepare to track the trend with respect to the model parameters.
      final double[][] trend = new double[observations.length][3];

      // Set the initial guess for the trend.
      trend[0][0] = estimatedTrend(observations);

      final double[][] jacobian = new double[observations.length][2];

      // The first elements of the Jacobian are zero as they are not dependent
      // upon the dampening factors, and so are the initial derivatives of the
      // trend which respect to those factors.
      jacobian[0][0] = jacobian[0][1] = trend[0][1] = trend[0][2] = 0.0;

      // Calculate the rest of the Jacobian using the current observation
      // and the immediately previous prediction.
      for (int i = 1; i < jacobian.length; ++i)
      {
        // Set the Jacobian with respect to alpha.
        jacobian[i][0] = observations[i] - smoothed[i - 1] - trend[i - 1][0] + (1 - alpha) * (jacobian[i - 1][0] + trend[i - 1][1]);

        // Set the Jacobian with respect to beta.
        jacobian[i][1] = (1 - alpha) * (jacobian[i - 1][1] + trend[i - 1][2]);

        // Adjust the trend.
        trend[i][0] = beta * (smoothed[i] - smoothed[i - 1]) + (1 - beta) * trend[i - 1][0];

        // Adjust the derivative of the trend with respect to alpha.
        trend[i][1] = beta * (jacobian[i][0] - jacobian[i - 1][0]) + (1 - beta) * trend[i - 1][1];

        // Adjust the derivative of the trend with respect to beta.
        trend[i][2] = smoothed[i] - smoothed[i - 1] - trend[i - 1][0] + beta * (jacobian[i][1] - jacobian[i - 1][1]) + (1 - beta) * trend[i - 1][2];
      }

      return jacobian;
    };
  }

  /**
   * Gets a validator that ensures that the values of \(\alpha\) and
   * \(\beta\) as determined by the optimization algorithm remain within
   * their expected bounds during all iterations.
   *
   * @return A {@link ParameterValidator}.
   */
  private ParameterValidator optimizationParameterValidator()
  {
    return points -> {
      final double alpha = points.getEntry(0) < MIN_DAMPENING_FACTOR
                           // If alpha is below the lower threshold, reset
                           // it to the lower threshold.
                           ? MIN_DAMPENING_FACTOR
                           : points.getEntry(0) > MAX_DAMPENING_FACTOR
                             // If alpha is above the upper threshold,
                             // reset it to the upper threshold.
                             ? MAX_DAMPENING_FACTOR
                             // If alpha is within its bounds, use its
                             // calculated value.
                             : points.getEntry(0);

      final double beta = points.getEntry(1) < MIN_DAMPENING_FACTOR
                          // If beta is below the lower threshold, reset
                          // it to the lower threshold.
                          ? MIN_DAMPENING_FACTOR
                          : points.getEntry(1) > MAX_DAMPENING_FACTOR
                            // If beta is above the upper threshold,
                            // reset it to the upper threshold.
                            ? MAX_DAMPENING_FACTOR
                            // If beta is within its bounds, use its
                            // calculated value.
                            : points.getEntry(1);

      return new ArrayRealVector(new double[] { alpha, beta });
    };
  }

  /**
   * Smoothens a collection of observations by exponentially dampening them
   * using factors \(\alpha\) and \(\beta\).
   *
   * @param observations The observations to smoothen.
   * @param alpha        The dampening factor \(\alpha\).
   * @param beta         The dampening factor \(\beta\).
   * @return A two-dimension array where the first dimension contains the
   * smoothened observations corresponding to the observations, and the second
   * dimension contains an estimate of the trend at each of the observed
   * points.
   */
  private double[][] smoothenObservations(final double[] observations, final double alpha, final double beta)
  {
    final double[][] smoothed = new double[2][observations.length];

    // Use the first observation as the first smooth observation.
    smoothed[0][0] = observations[0];

    // Estimate the overall trend using the first and last observations.
    smoothed[1][0] = estimatedTrend(observations);

    // Generate the rest using the smoothing formula.
    for (int i = 1; i < observations.length; ++i)
    {
      smoothed[0][i] = alpha * observations[i] + (1 - alpha) * (smoothed[0][i - 1] + smoothed[1][i - 1]);

      // Update the trend.
      smoothed[1][i] = beta * (smoothed[0][i] - smoothed[0][i - 1]) + (1 - beta) * smoothed[1][i - 1];
    }

    return smoothed;
  }
}
