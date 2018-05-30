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
 * Generates forecast for a sample that shows seasonality (periodicity) in
 * addition to upward and/or downward trends, using triple exponential
 * smoothing, where an observation \(\bf O\) is dampened exponentially to
 * get a smooth version \(\bf S\) as
 * </p>
 *
 * <p>
 * <br>
 * \(\large \boxed{S_i = \alpha(O_i - P_{i-m}) + (1 - \alpha)(S_{i-1} + T_{i-1})}\),
 * <br>
 * \(\large \boxed{T_i = \beta(S_i - S_{i-1}) + (1 - \beta)T_{i-1}}\), and
 * <br>
 * \(\large \boxed{P_i = \gamma(O_i - S_{i-1} - T_{i-1}) + (1 - \gamma)P_{i-m}}\)
 * </p>
 *
 * <p>
 * where, \(\bf i\) is an index that ranges from {@literal 1} to the number of
 * observations in the sample, \(\bf m\) is the number of observations in a
 * single season, \(\bf O_i\) is the {@literal i-th} observation, \(\bf S_i\)
 * its smooth version, \(\bf \alpha\), \(\bf \beta\) and \(\bf \gamma\) are
 * dampening factors between \(\bf 0.0\) and \(\bf 1.0\) responsible for
 * smoothing out the observations, \(\bf T_i\) is an estimate of the upward
 * or downward trend for the {@literal i-th} observation, and \(\bf P_i\) an
 * estimate of seasonality, known as a seasonality index. This means
 * </p>
 *
 * <p>
 * <br>
 * \(\large S_{m+1} = \alpha(O_{m+1} - P_1) + (1 - \alpha)(S_m + T_m)\)
 * <br>
 * \(\large S_{m+2} = \alpha(O_{m+2} - P_2) + (1 - \alpha)(S_{m+1} + T_{m+1})\)
 * <br>
 * \(\large S_{m+3} = \alpha(O_{m+3} - P_3) + (1 - \alpha)(S_{m+2} + T_{m+2})\)
 * <br>
 * ... and so on, and
 * </p>
 *
 * <p>
 * <br>
 * \(\large T_{m+1} = \beta(S_{m+1} - S_m) + (1 - \beta)T_m\)
 * <br>
 * \(\large T_{m+2} = \beta(S_{m+2} - S_{m+1}) + (1 - \beta)T_{m+1}\)
 * <br>
 * \(\large T_{m+3} = \beta(S_{m+3} - S_{m+2}) + (1 - \beta)T_{m+2}\)
 * <br>
 * ... and so on.
 * </p>
 *
 * <p>
 * This version is known as <i>additive triple exponential smoothing</i> due to
 * the fact that the effect of seasonality is <i>added</i> to the observed
 * value (\(S_{m+1} = \alpha(O_{m+1} - P_1) + ...\)) when smoothing the
 * observations.
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
 * Given that \(S_i\) depends on \(P_{i-m}\), the smooth values cannot be
 * calculated for observations up to \(O_m\) (since \(i-m\) is negative and the
 * corresponding \(P_{i-m}\) are unavailable). Therefore, the smooth values
 * \(S_1\) to \(S_m\) are chosen upfront as
 * </p>
 *
 * <p>
 * <br>
 * \(\large S_1 = O_1\)
 * <br>
 * \(\large S_2 = O_2\)
 * <br>
 * ...
 * <br>
 * \(\large S_m = O_m\)
 * </p>
 *
 * <p>
 * The initial estimate for the trend, \(T_1\) is calculated as
 * </p>
 *
 * <p>
 * <br>
 * \(\large \boxed{T_1 = \frac{1}{m^2}[(O_{m+1} - O_1) + (O_{m+2} - O_2) + ... + (O_{2m} - O_m)]}\)
 * </p>
 *
 * <p>
 * Given the dependence of the seasonality index \(P_i\) on \(P_{i-m}\), the
 * first \(m\) indices \(P_1\) to \(P_m\) also need to be estimated. For the
 * model to produce an optimum forecast, this requires a slightly involved
 * procedure, as explained below.
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
 * \(\large A_1 = \frac{1}{m}(O_1 + O_2 + ... + O_m)\)
 * <br>
 * \(\large A_2 = \frac{1}{m}(O_{m+1} + O_{m+2} + ... + O_{2m})\)
 * <br>
 * ...
 * <br>
 * \(\large A_n = \frac{1}{m}(O_{(n-1)m+1} + O_{(n-1)m+2} + ... + O_{nm})\)
 * </p>
 *
 * <p>
 * <b>Step 3:</b> Calculate the first \(m\) seasonality indices \(P_1\) to
 * \(P_m\) as
 * </p>
 *
 * <p>
 * <br>
 * \(\large P_1 = \frac{1}{n}(\frac{O_1}{A_1} + \frac{O_{m+1}}{A_2} + ... + \frac{O_{(n-1)m+1}}{A_n})\)
 * <br>
 * \(\large P_2 = \frac{1}{n}(\frac{O_2}{A_1} + \frac{O_{m+2}}{A_2} + ... + \frac{O_{(n-1)m+2}}{A_n})\)
 * <br>
 * ...
 * <br>
 * \(\large P_m = \frac{1}{n}(\frac{O_m}{A_1} + \frac{O_{2m}}{A_2} + ... + \frac{O_{nm}}{A_n})\)
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
  private final int periodsPerSeason;

  /**
   * Creates a model with a specified number of periods per season in the
   * sample.
   *
   * @param periodsPerSeason The number of periods per season for the model.
   * @throws IllegalArgumentException if {@code periodsPerSeason} is
   *                                  {@literal zero} or negative.
   */
  public TripleExponentialSmoothingForecastModel(final int periodsPerSeason)
  {
    if (periodsPerSeason < 1)
    {
      throw new IllegalArgumentException("Periods per season must be greater than zero.");
    }

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

    // Smoothen the observations using the optimal values for alpha, beta and gamma.
    final double[][] processed = smoothenObservations(observations, optimalDampeningFactors[0], optimalDampeningFactors[1], optimalDampeningFactors[2]);
    final double[] seasonality = processed[2], smoothed = processed[0], trend = processed[1];

    // Add the smooth observations as the predictions for the known
    // observations.
    final double[] predictions = Arrays.copyOf(smoothed, samples + projections);

    // Add specified number of predictions beyond the sample.
    for (int i = 0; i < projections; ++i)
    {
      // For predictions beyond the available sample, the baseline period
      // should be within the last season. Determine the baseline period
      // for the current prediction.
      final int period = samples - periodsPerSeason + i % periodsPerSeason - 1;

      predictions[samples + i] = smoothed[samples - 1] + (i + 1) * trend[samples - 1] + seasonality[period];
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
   * \(\large T_1 = \frac{1}{m^2}[(O_{m+1} - O_1) + (O_{m+2} - O_2) + ... + (O_{2m} - O_m)]\)
   * <br>
   * </p>
   *
   * <p>
   * where, \(m\) is the number of observations in a single season and
   * \(O_1\) is the first (chronologically oldest) observation.
   * </p>
   *
   * @param observations The observations for which the trend is required.
   * @return An estimate for the initial trend for the specified observations.
   */
  private double estimatedTrend(final double[] observations)
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
   * guess for the dampening factors \(\alpha\), \(\beta\) and \(\gamma\).
   * </p>
   *
   * <p>
   * The optimization algorithm requires the following input:
   * </p>
   *
   * <ol>
   * <li>The observations for which \(\alpha\), \(\beta\) and \(\gamma\)
   * need to be optimized;</li>
   * <li>An initial guess for \(\alpha\), \(\beta\) and \(\gamma\);</li>
   * <li>A function that can take the observations and some values for
   * \(\alpha\), \(\beta\) and \(\gamma\), and produce predictions for each
   * of the observations with those values of the dampening factors. This is
   * known as the model function for the algorithm. The algorithm internally
   * calculates the MSE for each set of values of the dampening factors by
   * using the given observations and their corresponding predictions for the
   * specified values of \(\alpha\), \(beta\) and \(\gamma\);</li>
   * <li>A function that can take the observations and some values for
   * \(\alpha\), \(\beta\) and \(\gamma\), and produce a <i>Jacobian</i> for
   * each prediction made by the model function. The <i>Jacobian</i> is a
   * linear approximation of the derivative of a predicted value with respect
   * to the variable parameters \(\alpha\), \(\beta\) and \(\gamma\) and hence
   * serves to determine whether the trend at each predicted value matches that
   * of the corresponding observed value. This information is critical in
   * optimizing the values of \(\alpha\), \(beta\) and \(\gamma\) as the
   * derivative determines whether the values are too high or too low, and
   * therefore whether they need to be lowered or raised;</li>
   * <li>A function that can validate whether a specific value of
   * \(\alpha\), \(\beta\) or \(\gamma\) is within the bounds of the
   * problem-space. For the purposes of exponential smoothing, \(\alpha\),
   * \(\beta\) and \(\gamma\) must be between {@literal 0.1} and
   * {@literal 0.9}.</li>
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
        .start(new double[] { INITIAL_DAMPENING_FACTOR, INITIAL_DAMPENING_FACTOR, INITIAL_DAMPENING_FACTOR })
        .build();
  }

  /**
   * Gets the model function to use for optimizing the values of \(\alpha\),
   * \(\beta\) and \(\gamma\), which is just the predicted value for each
   * observed value. The optimization algorithm then uses the difference
   * between the observed and corresponding predicted values to find the
   * optimal values for \(\alpha\), \(\beta\) and \(\gamma\).
   *
   * @param observations The observations for which optimal values of
   *                     \(\alpha\), \(\beta\) and \(\gamma\) are required.
   * @return A {@link MultivariateVectorFunction}.
   */
  private MultivariateVectorFunction optimizationModelFunction(final double[] observations)
  {
    return params -> smoothenObservations(observations, params[0], params[1], params[2])[0];
  }

  /**
   * <p>
   * Gets the <i>Jacobian</i> (\(J\)) corresponding to the model function used
   * for optimizing the values of \(\alpha\), \(\beta\) and \(\gamma\). The
   * <i>Jacobian</i> for a function \(S\) of \(k\) parameters \(x_k\) is a
   * matrix, where an element \(J_{ik}\) of the matrix is given by
   * \(J_{ik} = \frac{\partial S_i}{\partial x_k}\), \(x_k\) are the
   * \(k\) parameters on which the function \(S\) is dependent, and \(S_i\) are
   * the values of the function \(S\) at \(i\) distinct points. In the case of
   * the triple exponential smoothing forecast model, there are three
   * parameters - \(\alpha\), \(\beta\) and \(\gamma\) that impact the predicted
   * value for a given observed value. Therefore, (\(J\)) contains three values
   * for every \(S_i\)
   * </p>
   *
   * <p>
   * <br>
   * \(\large J_{i\alpha}\), defined as \(\boxed{J_{i\alpha} = \frac{\partial S_i}{\partial \alpha}}\),
   * <br>
   * \(\large J_{i\beta}\), defined as \(\boxed{J_{i\beta} = \frac{\partial S_i}{\partial \beta}}\), and
   * <br>
   * \(\large J_{i\gamma}\), defined as \(\boxed{J_{i\gamma} = \frac{\partial S_i}{\partial \gamma}}\), or
   * <br>
   * </p>
   *
   * <p>
   * \(\large J = \begin{bmatrix}
   * J_{1\alpha} &amp; J_{2\alpha} &amp; ... &amp; J_{n\alpha}
   * \\
   * J_{1\beta} &amp; J_{2\beta} &amp; ... &amp; J_{n\beta}
   * \\
   * J_{1\gamma} &amp; J_{2\gamma} &amp; ... &amp; J_{n\gamma}
   * \end{bmatrix}
   * = \begin{bmatrix}
   * \frac{\partial S_1}{\partial \alpha} &amp; \frac{\partial S_2}{\partial \alpha} &amp; ... &amp; \frac{\partial S_n}{\partial \alpha}
   * \\
   * \frac{\partial S_1}{\partial \beta} &amp; \frac{\partial S_2}{\partial \beta} &amp; ... &amp; \frac{\partial S_n}{\partial \beta}
   * \\
   * \frac{\partial S_1}{\partial \gamma} &amp; \frac{\partial S_2}{\partial \gamma} &amp; ... &amp; \frac{\partial S_n}{\partial \gamma}
   * \end{bmatrix}\)
   * </p>
   *
   * <p>
   * For the triple exponential smoothing model, the function \(S\) is
   * defined as (see above)
   * </p>
   *
   * <p>
   * <br>
   * \(\large S_i = \alpha(O_i - P_{i-m}) + (1 - \alpha)(S_{i-1} + T_{i-1})\),
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
   * <br> becomes
   * <br><br>
   * \(\large J_{i\alpha} = \frac{\partial}{\partial \alpha}[\alpha{(O_i - P_{i-m})} + (1 - \alpha)(S_{i-1} + T_{i-1})]\),
   * <br> or (by the associative rule of differentiation)
   * <br><br>
   * \(\large J_{i\alpha} = \frac{\partial}{\partial \alpha}(\alpha{O_i})
   * - \frac{\partial}{\partial \alpha}(\alpha{P_{i-m}})
   * + \frac{\partial}{\partial \alpha}[(1 - \alpha)(S_{i-1} + T_{i-1})]\),
   * <br> or (by the chain rule of differentiation)
   * <br><br>
   * \(\large J_{i\alpha} = \alpha\frac{\partial O_i}{\partial \alpha} + O_i\frac{\partial \alpha}{\partial \alpha}
   * - \alpha\frac{\partial P_{i-m}}{\partial \alpha} - P_{i-m}\frac{\partial \alpha}{\partial \alpha}
   * + (1 - \alpha)\frac{\partial}{\partial \alpha}(S_{i-1} + T_{i-1}) + (S_{i-1} + T_{i-1})\frac{\partial}{\partial \alpha}(1 - \alpha)\),
   * <br> or (since \(\frac{\partial O_i}{\partial \alpha} = 0\), given that
   * \(O_i\) does not depend on \(\alpha\))
   * <br><br>
   * \(\large J_{i\alpha} = O_i
   * - \alpha\frac{\partial P_{i-m}}{\partial \alpha} - P_{i-m}
   * + (1 - \alpha)(\frac{\partial S_{i-1}}{\partial \alpha} + \frac{\partial T_{i-1}}{\partial \alpha}) - (S_{i-1} + T_{i-1})\)
   * <br> or (upon rearrangement of terms)
   * <br><br>
   * \(\large \boxed{J_{i\alpha} = O_i - S_{i-1} - T_{i-1} - P_{i-m}
   * - \alpha\frac{\partial P_{i-m}}{\partial \alpha}
   * + (1 - \alpha)(\frac{\partial S_{i-1}}{\partial \alpha} + \frac{\partial T_{i-1}}{\partial \alpha})}\)
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
   * becomes
   * <br><br>
   * \(\large J_{i\beta} = \frac{\partial}{\partial \beta}[\alpha(O_i - P_{i-m}) + (1 - \alpha)(S_{i-1} + T_{i-1})]\),
   * <br> or (by the associative rule of differentiation)
   * <br><br>
   * \(\large J_{i\beta} = \alpha\frac{\partial}{\partial \beta}(O_i - P_{i-m}) + (1 - \alpha)\frac{\partial}{\partial \beta}(S_{i-1} + T_{i-1})]\),
   * <br> or (by the associative rule of differentiation)
   * <br><br>
   * \(\large \boxed{J_{i\beta} = (1 - \alpha)(\frac{\partial S_{i-1}}{\partial \beta} + \frac{\partial T_{i-1}}{\partial \beta})
   * - \alpha\frac{\partial P_{i-m}}{\partial \beta}}\).
   * </p>
   *
   * <p>
   * Continuing in a similar fashion,
   * </p>
   *
   * <p>
   * <br>
   * \(\large J_{i\gamma} = \frac{\partial S_i}{\partial \gamma}\)
   * <br>
   * becomes
   * <br><br>
   * \(\large \boxed{J_{i\gamma} = (1 - \alpha)(\frac{\partial S_{i-1}}{\partial \gamma} + \frac{\partial T_{i-1}}{\partial \gamma})
   * - \alpha\frac{\partial P_{i-m}}{\partial \gamma}}\).
   * </p>
   *
   * <p>
   * Given that \(J_{i\alpha}\), \(J_{i\beta}\) and \(J_{i\gamma}\) depend on
   * \(\frac{\partial T_i}{\partial \alpha}\),
   * \(\frac{\partial T_i}{\partial \beta}\),
   * \(\frac{\partial T_i}{\partial \gamma}\),
   * \(\frac{\partial P_i}{\partial \alpha}\),
   * \(\frac{\partial P_i}{\partial \beta}\), and
   * \(\frac{\partial P_i}{\partial \gamma}\), these need to be
   * computed as well.
   * </p>
   *
   * <p>
   * <br>
   * \(\large \boxed{\frac{\partial T_i}{\partial \alpha} = \beta(\frac{\partial S_i}{\partial \alpha}
   * - \frac{\partial S_{i-1}}{\partial \alpha})
   * + (1 - \beta)\frac{\partial T_{i-1}}{\partial \alpha}}\),
   * <br>
   * \(\large \boxed{\frac{\partial T_i}{\partial \beta} = S_i - S_{i-1} - T_{i-1}
   * + \beta(\frac{\partial S_i}{\partial \beta} - \frac{\partial S_{i-1}}{\partial \beta})
   * + (1 - \beta)\frac{\partial T_{i-1}}{\partial \beta}}\),
   * <br>
   * \(\large \boxed{\frac{\partial T_i}{\partial \gamma} = \beta(\frac{\partial S_i}{\partial \gamma}
   * - \frac{\partial S_{i-1}}{\partial \gamma})
   * + (1 - \beta)\frac{\partial T_{i-1}}{\partial \gamma}}\),
   * <br><br>
   * \(\large \boxed{\frac{\partial P_i}{\partial \alpha} = (1 - \gamma)\frac{\partial P_{i-m}}{\partial \alpha}
   * - \gamma(\frac{\partial S_{i-1}}{\partial \alpha} + \frac{\partial T_{i-1}}{\partial \alpha})}\),
   * <br>
   * \(\large \boxed{\frac{\partial P_i}{\partial \beta} = (1 - \gamma)\frac{\partial P_{i-m}}{\partial \beta}
   * - \gamma(\frac{\partial S_{i-1}}{\partial \beta} + \frac{\partial T_{i-1}}{\partial \beta})}\),
   * <br>
   * \(\large \boxed{\frac{\partial P_i}{\partial \gamma} = O_i - S_{i-1} - T_{i-1} - P_{i-m}
   * - \gamma(\frac{\partial S_{i-1}}{\partial \gamma} + \frac{\partial T_{i-1}}{\partial \gamma})
   * + (1 - \gamma)\frac{\partial P_{i-m}}{\partial \gamma}}\).
   * </p>
   *
   * <p>
   * Given that the smooth observations \(S_1\) to \(S_m\) have no dependence
   * on \(\alpha\), \(\beta\) or \(\gamma\),
   * \(\boxed{\frac{\partial S_1}{\partial \alpha} = \frac{\partial S_1}{\partial \beta} = \frac{\partial S_1}{\partial \gamma}
   * = \frac{\partial S_2}{\partial \alpha} = \frac{\partial S_2}{\partial \beta} = \frac{\partial S_2}{\partial \gamma}
   * = ...
   * = \frac{\partial S_m}{\partial \alpha} = \frac{\partial S_m}{\partial \beta} = \frac{\partial S_m}{\partial \gamma}
   * = 0}\).
   * </p>
   *
   * <p>
   * Similarly, since the initial estimate for the trend, \(T_1\) has no
   * dependence on \(\alpha\), \(\beta\) or \(\gamma\),
   * \(\boxed{\frac{\partial T_1}{\partial \alpha} = \frac{\partial T_1}{\partial \beta} = \frac{\partial T_1}{\partial \gamma} = 0}\).
   * </p>
   *
   * <p>
   * In the same way, since the initial seasonality indices \(P_1\) to \(P_m\)
   * have no dependence on \(\alpha\), \(\beta\) or \(\gamma\),
   * \(\boxed{\frac{\partial P_1}{\partial \alpha} = \frac{\partial P_1}{\partial \beta} = \frac{\partial P_1}{\partial \gamma}
   * = \frac{\partial P_2}{\partial \alpha} = \frac{\partial P_2}{\partial \beta} = \frac{\partial P_2}{\partial \gamma}
   * = ...
   * = \frac{\partial P_m}{\partial \alpha} = \frac{\partial P_m}{\partial \beta} = \frac{\partial P_m}{\partial \gamma}
   * = 0}\).
   * </p>
   *
   * <p>
   * These initial values, combined with the recursive formulas above are used
   * to determine the remaining values for the Jacobian.
   * </p>
   *
   * @param observations The observations \(O_i\) for which optimal values of
   *                     \(\alpha\), \(\beta\) and \(\gamma\) are required.
   * @return A {@link MultivariateMatrixFunction}, which is a three-column
   * matrix whose first column contains elements corresponding to the
   * <i>Jacobian</i> for the model function with respect to the parameter
   * \(\alpha\), and the second column contains elements corresponding to the
   * <i>Jacobian</i> for the model function with respect to the parameter
   * \(\beta\), and the third column contains elements corresponding to the
   * <i>Jacobian</i> for the model function with respect to the parameter
   * \(\gamma\).
   * @see <a href="https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm">Levenberg-Marquardt algorithm</a>
   * @see <a href="https://en.wikipedia.org/wiki/Least_squares">Least squares</a>
   * @see <a href="https://en.wikipedia.org/wiki/Curve_fitting">Curve fitting</a>
   * @see <a href="https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant">Jacobian matrix</a>
   */
  private MultivariateMatrixFunction optimizationModelJacobian(final double[] observations)
  {
    return params -> {
      final double alpha = params[0], beta = params[1], gamma = params[2];

      // Determine the sample size.
      final int samples = observations.length;

      // Smoothen the observations.
      final double[][] processed = smoothenObservations(observations, alpha, beta, gamma);
      final double[] seasonality = processed[2], smoothed = processed[0], trend = processed[1];

      // Prepare to track the changes in trend with respect to the model
      // parameters.
      final double[] dTA = new double[samples], dTB = new double[samples], dTG = new double[samples];

      // Prepare to track the changes in seasonality with respect to the model
      // parameters.
      final double[] dSA = new double[samples], dSB = new double[samples], dSG = new double[samples];

      final double[][] jacobian = new double[samples][3];

      for (int i = 0; i < samples; ++i)
      {
        if (i < periodsPerSeason)
        {
          // For the first season, the smoothed observations and seasonality
          // indices do not depend upon the model parameters, and are hence
          // their derivatives with respect to the model parameters are zero.
          jacobian[i][0] = jacobian[i][1] = jacobian[i][2] = dSA[i] = dSB[i] = dSG[i] = 0.0;
        }
        else
        {
          // Set the Jacobian with respect to alpha.
          jacobian[i][0] = observations[i] - smoothed[i - 1] - trend[i - 1] - seasonality[i - periodsPerSeason]
              - alpha * dSA[i - periodsPerSeason]
              + (1 - alpha) * (jacobian[i - 1][0] + dTA[i - 1]);

          // Set the Jacobian with respect to beta.
          jacobian[i][1] = (1 - alpha) * (jacobian[i - 1][1] - dTB[i - 1]) - alpha * dSB[i - periodsPerSeason];

          // Set the Jacobian with respect to gamma.
          jacobian[i][2] = (1 - alpha) * (jacobian[i - 1][2] - dTG[i - 1]) - alpha * dSG[i - periodsPerSeason];

          // Calculate the derivative of the seasonality index with respect
          // to alpha.
          dSA[i] = (1 - gamma) * dSA[i - periodsPerSeason] - gamma * (jacobian[i - 1][0] + dTA[i - 1]);

          // Calculate the derivative of the seasonality index with respect
          // to beta.
          dSB[i] = (1 - gamma) * dSB[i - periodsPerSeason] - gamma * (jacobian[i - 1][1] + dTB[i - 1]);

          // Calculate the derivative of the seasonality index with respect
          // to gamma.
          dSG[i] = observations[i] - smoothed[i - 1] - trend[i - 1] - seasonality[i - periodsPerSeason]
              + (1 - gamma) * dSG[i - periodsPerSeason]
              - gamma * (jacobian[i - 1][2] + dTG[i - 1]);
        }

        if (i == 0)
        {
          // The initial estimate for the trend does not depend upon the
          // model parameters, and hence its derivative with respect to the
          // model parameters is zero.
          dTA[i] = dTB[i] = dTG[i] = 0.0;
        }
        else
        {
          // Adjust the derivative of the trend with respect to alpha.
          dTA[i] = beta * (jacobian[i][0] - jacobian[i - 1][0]) + (1 - beta) * dTA[i - 1];

          // Adjust the derivative of the trend with respect to beta.
          dTB[i] = smoothed[i] - smoothed[i - 1] - trend[i - 1] + beta * (jacobian[i][1] - jacobian[i - 1][1]) + (1 - beta) * dTB[i - 1];

          // Adjust the derivative of the trend with respect to gamma.
          dTG[i] = beta * (jacobian[i][2] - jacobian[i - 1][2]) + (1 - beta) * dTG[i - 1];
        }
      }

      return jacobian;
    };
  }

  /**
   * Gets a validator that ensures that the values of \(\alpha\), \(\beta\)
   * and \(\gamma\) as determined by the optimization algorithm remain within
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

      final double gamma = points.getEntry(2) < MIN_DAMPENING_FACTOR
                           // If gamma is below the lower threshold, reset
                           // it to the lower threshold.
                           ? MIN_DAMPENING_FACTOR
                           : points.getEntry(2) > MAX_DAMPENING_FACTOR
                             // If gamma is above the upper threshold,
                             // reset it to the upper threshold.
                             ? MAX_DAMPENING_FACTOR
                             // If gamma is within its bounds, use its
                             // calculated value.
                             : points.getEntry(2);

      return new ArrayRealVector(new double[] { alpha, beta, gamma });
    };
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
   * @return A three-dimension array where the first dimension contains the
   * smoothened observations corresponding to the observations, the second
   * dimension contains an estimate of the trend at each of the observed
   * points, and the third an estimate of the seasonality at each observed
   * point.
   */
  private double[][] smoothenObservations(final double[] observations, final double alpha, final double beta, final double gamma)
  {
    // Determine the sample size and the number of seasons for the sample.
    final int samples = observations.length, seasons = samples / periodsPerSeason;

    final double[] seasonality = new double[samples], smoothed = new double[samples], trend = new double[samples];

    // Determine seasonal averages, as they are required to calculate the
    // initiate estimates for seasonality.
    final double[] seasonalAverages = seasonalAverages(observations);

    for (int i = 0; i < samples; ++i)
    {
      if (i < periodsPerSeason)
      {
        // For the first season, set the observations as the smoothed values,
        // because the seasonality indices required to calculate the smoothed
        // versions are not available.
        smoothed[i] = observations[i];
      }
      else
      {
        // Calculate the smoothed version.
        smoothed[i] = alpha * (observations[i] - seasonality[i - periodsPerSeason]) + (1 - alpha) * (smoothed[i - 1] + trend[i - 1]);
      }

      if (i == 0)
      {
        // For the initial value of the trend, use an estimate based on the
        // overall trend for the observations.
        trend[i] = estimatedTrend(observations);
      }
      else
      {
        // Calculate the trend.
        trend[i] = beta * (smoothed[i] - smoothed[i - 1]) + (1 - beta) * trend[i - 1];
      }

      if (i < periodsPerSeason)
      {
        // For the first season, estimate the seasonality indices.
        for (int j = 0; j < seasons; ++j)
        {
          seasonality[i] += observations[i + j * periodsPerSeason] / seasonalAverages[j];
        }

        seasonality[i] /= seasons;
      }
      else
      {
        // Calculate the seasonality index.
        seasonality[i] = gamma * (observations[i] - smoothed[i - 1] - trend[i - 1]) + (1 - gamma) * seasonality[i - 1];
      }
    }

    return new double[][] { smoothed, trend, seasonality };
  }
}
