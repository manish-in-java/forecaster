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
    final double[][] processed = smoothenObservations(observations, optimalDampeningFactors[0], optimalDampeningFactors[1]);
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
   * calculates the MSE for each set of values of the dampening factors by
   * using the given observations and their corresponding predictions for the
   * specified values of \(\alpha\) and \(beta\);</li>
   * <li>A function that can take the observations and some values for
   * \(\alpha\) and \(\beta\), and produce a <i>Jacobian</i> for each
   * prediction made by the model function. The <i>Jacobian</i> is a linear
   * approximation of the derivative of a predicted value with respect
   * to the variable parameters \(\alpha\) and \(\beta\) and hence serves to
   * determine whether the trend at each predicted value matches that of the
   * corresponding observed value. This information is critical in optimizing
   * the values of \(\alpha\) and \(beta\) as the derivative determines
   * whether the values are too high or too low, and therefore whether
   * they need to be lowered or raised;</li>
   * <li>A function that can validate whether a specific value of
   * \(\alpha\) or \(\beta\) is within the bounds of the problem-space. For
   * the purposes of exponential smoothing, \(\alpha\) and \(\beta\) must be
   * between {@literal 0.0} and {@literal 1.0}.</li>
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
   * Gets the <i>Jacobian</i> (\(j\)) corresponding to the model function used
   * for optimizing the values of \(\alpha\) and \(\beta\). The <i>Jacobian</i>
   * for a function \(l\) of \(k\) parameters \(x_k\) is a matrix, where an
   * element \(j_{tk}\) of the matrix is given by
   * \(j_{tk} = \frac{\partial l_t}{\partial x_k}\), \(x_k\) are the
   * \(k\) parameters on which the function \(l\) is dependent, and \(l_t\) are
   * the values of the function \(l\) at \(t\) distinct points. In the case of
   * the double exponential smoothing forecast model, there are two parameters
   * - \(\alpha\) and \(beta\) that impact the predicted value for a given
   * observed value. Therefore, (\(j\)) contains two values for every \(l_t\)
   * </p>
   *
   * <p>
   * <br>
   * \(\large j_{t\alpha}\), defined as \(\boxed{j_{t\alpha} = \frac{\partial l_t}{\partial \alpha}}\), and
   * <br>
   * \(\large j_{t\beta}\), defined as \(\boxed{j_{t\beta} = \frac{\partial l_t}{\partial \beta}}\), or
   * <br>
   * </p>
   *
   * <p>
   * \(\large j = \begin{bmatrix}
   * j_{1\alpha} &amp; j_{2\alpha} &amp; ... &amp; j_{n\alpha}
   * \\
   * j_{1\beta} &amp; j_{2\beta} &amp; ... &amp; j_{n\beta}
   * \end{bmatrix}
   * = \begin{bmatrix}
   * \frac{\partial l_1}{\partial \alpha} &amp; \frac{\partial l_2}{\partial \alpha} &amp; ... &amp; \frac{\partial l_n}{\partial \alpha}
   * \\
   * \frac{\partial l_1}{\partial \beta} &amp; \frac{\partial l_2}{\partial \beta} &amp; ... &amp; \frac{\partial l_n}{\partial \beta}
   * \end{bmatrix}\)
   * </p>
   *
   * <p>
   * For the double exponential smoothing model, the function \(l\) is
   * defined as (see above)
   * </p>
   *
   * <p>
   * <br>
   * \(\large l_t = \alpha{y_t} + (1 - \alpha)(l_{t-1} + b_{t-1})\)
   * <br>
   * </p>
   *
   * <p>
   * Therefore,
   * </p>
   *
   * <p>
   * <br>
   * \(\large j_{t\alpha} = \frac{\partial l_t}{\partial \alpha}\)
   * <br>
   * becomes (through the rules for partial derivatives and the chain rule of
   * differentiation)
   * <br><br>
   * \(\large j_{t\alpha} = \alpha{\frac{\partial y_t}{\partial \alpha}} + y_t{\frac{\partial \alpha}{\partial \alpha}}
   * + (1 - \alpha)(\frac{\partial S{i-1}}{\partial \alpha} + \frac{\partial T{i-1}}{\partial \alpha})
   * - (l_{t-1} + b_{t-1}){\frac{\partial \alpha}{\partial \alpha}}\),
   * <br>
   * or (since \(\frac{\partial y_t}{\partial \alpha} = 0\), given that \(y_t\)
   * does not depend on \(\alpha\))
   * <br><br>
   * \(\large \boxed{j_{t\alpha} = y_t - l_{t-1} - b_{t-1} + (1 - \alpha)(\frac{\partial l_{t-1}}{\partial \alpha} + \frac{\partial b_{t-1}}{\partial \alpha})}\)
   * </p>
   *
   * <p>
   * Similarly,
   * </p>
   *
   * <p>
   * <br>
   * \(\large j_{t\beta} = \frac{\partial l_t}{\partial \beta}\)
   * <br>
   * becomes (since \(\frac{\partial y_t}{\partial \beta} = \frac{\partial \alpha}{\partial \beta} = 0\),
   * given that \(y_t\) and \(\alpha\) do not depend on \(\beta\))
   * <br><br>
   * \(\large \boxed{j_{t\beta} = (1 - \alpha)(\frac{\partial l_{t-1}}{\partial \beta} + \frac{\partial b_{t-1}}{\partial \beta})}\)
   * </p>
   *
   * <p>
   * Given that \(j_{t\alpha}\) and \(j_{t\beta}\) depend on
   * \(\frac{\partial b_{t-1}}{\partial \alpha}\) and
   * \(\frac{\partial b_{t-1}}{\partial \beta}\) as well
   * </p>
   *
   * <p>
   * <br>
   * \(\large \boxed{\frac{\partial b_t}{\partial \alpha} = \beta(\frac{\partial l_t}{\partial \alpha} - \frac{\partial l_{t-1}}{\partial \alpha})
   * + (1 - \beta)\frac{\partial b_{t-1}}{\partial \alpha}}\), and
   * </p>
   *
   * <p>
   * <br>
   * \(\large \frac{\partial b_t}{\partial \beta} = \beta(\frac{\partial l_t}{\partial \beta} - \frac{\partial l_{t-1}}{\partial \beta})
   * + (l_t - l_{t-1})\frac{\partial \beta}{\partial \beta}
   * + (1 - \beta)\frac{\partial b_{t-1}}{\partial \beta} - b_{t-1}\frac{\partial \beta}{\partial \beta}\), or
   * <br>
   * \(\large \boxed{\frac{\partial b_t}{\partial \beta} = l_t - l_{t-1} - b_{t-1}
   * + \beta(\frac{\partial l_t}{\partial \beta} - \frac{\partial l_{t-1}}{\partial \beta})
   * + (1 - \beta)\frac{\partial b_{t-1}}{\partial \beta}
   * }\)
   * </p>
   *
   * <p>
   * Due to the choice for \(l_1\) and \(b_1\), neither of which depends on
   * \(\alpha\) or \(\beta\),
   * </p>
   *
   * <p>
   * <br>
   * \(\large \boxed{\frac{\partial l_1}{\partial \alpha} =
   * \frac{\partial l_1}{\partial \beta} =
   * \frac{\partial b_1}{\partial \alpha} =
   * \frac{\partial b_1}{\partial \beta} =
   * 0}\)
   * </p>
   *
   * @param observations The observations \(y_t\) for which optimal values of
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
      final double[][] processed = smoothenObservations(observations, alpha, beta);
      final double[] smoothed = processed[0], trend = processed[1];

      // Determine the sample size.
      final int samples = observations.length;

      // Prepare to track the changes in trend with respect to the model
      // parameters.
      final double[] trendDerivativesAlpha = new double[samples], trendDerivativesBeta = new double[samples];

      final double[][] jacobian = new double[samples][2];

      // The first elements of the Jacobian are zero as they are not dependent
      // upon the dampening factors, and so are the initial derivatives of the
      // trend with respect to those factors.
      jacobian[0][0] = jacobian[0][1] = trendDerivativesAlpha[0] = trendDerivativesBeta[0] = 0.0;

      // Calculate the rest of the Jacobian using the current observation
      // and the immediately previous prediction.
      for (int i = 1; i < jacobian.length; ++i)
      {
        // Set the Jacobian with respect to alpha.
        jacobian[i][0] = observations[i] - smoothed[i - 1] - trend[i - 1] + (1 - alpha) * (jacobian[i - 1][0] + trendDerivativesAlpha[i - 1]);

        // Set the Jacobian with respect to beta.
        jacobian[i][1] = (1 - alpha) * (jacobian[i - 1][1] + trendDerivativesBeta[i - 1]);

        // Adjust the derivative of the trend with respect to alpha.
        trendDerivativesAlpha[i] = beta * (jacobian[i][0] - jacobian[i - 1][0]) + (1 - beta) * trendDerivativesAlpha[i - 1];

        // Adjust the derivative of the trend with respect to beta.
        trendDerivativesBeta[i] = smoothed[i] - smoothed[i - 1] - trend[i - 1] + beta * (jacobian[i][1] - jacobian[i - 1][1]) + (1 - beta) * trendDerivativesBeta[i - 1];
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
   * smoothened observations, and the second dimension contains an estimate of
   * the trend at each of the observed points.
   */
  private double[][] smoothenObservations(final double[] observations, final double alpha, final double beta)
  {
    // Determine the sample size.
    final int samples = observations.length;

    final double[] smoothed = new double[samples], trend = new double[samples];

    // Use the first observation as the first smooth observation.
    smoothed[0] = observations[0];

    // Estimate the overall trend using the first and last observations.
    trend[0] = estimatedInitialTrend(observations);

    // Generate the rest using the smoothing formula.
    for (int i = 1; i < samples; ++i)
    {
      smoothed[i] = alpha * observations[i] + (1 - alpha) * (smoothed[i - 1] + trend[i - 1]);

      // Update the trend.
      trend[i] = beta * (smoothed[i] - smoothed[i - 1]) + (1 - beta) * trend[i - 1];
    }

    return new double[][] { smoothed, trend };
  }
}
