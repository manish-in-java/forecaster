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
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.fitting.leastsquares.ParameterValidator;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.optim.SimpleVectorValueChecker;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * <p>
 * Generates forecast for a sample using exponential smoothing, where an
 * observation \(O\) is dampened exponentially to get a smooth version \(S\)
 * as
 * </p>
 *
 * <p>
 * <br>
 * \(S_i = \alpha{O_{i-1}} + (1 - \alpha)S_{i-1}\)
 * <br>
 * </p>
 *
 * <p>or its alternative form</p>
 *
 * <p>
 * <br>
 * \(S_i = S_{i-1} + \alpha({O_{i-1}} - S_{i-1})\)
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
 * \(S_2 = \alpha{O_1} + (1 - \alpha)S_1\)
 * <br>
 * \(S_3 = \alpha{O_2} + (1 - \alpha)S_2\)
 * <br>
 * \(S_4 = \alpha{O_3} + (1 - \alpha)S_3\)
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
 * This model, characterized by a dependence of \(S_i\) upon \(O_{i-1}\) was
 * originally proposed in {@literal 1986} by {@literal J. Stuart Hunter}.
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
 * @see <a href="https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc431.htm">Exponential Smoothing</a>
 */
public class SingleExponentialSmoothingForecastModel extends ForecastModel
{
  private static final double INITIAL_ALPHA                      = 0.5;
  private static final double MAX_ALPHA                          = 0.8;
  private static final int    MAX_ALPHA_OPTIMIZATION_EVALUATIONS = 100;
  private static final int    MAX_ALPHA_OPTIMIZATION_ITERATIONS  = 100;
  private static final double MIN_ALPHA                          = 0.2;

  private final AlphaOptimizer           alphaOptimizer;
  private final FirstPredictionGenerator firstPredictionGenerator;

  /**
   * Creates a model with simple average strategy for generating the first
   * prediction and a {@literal non-linear least-squares} optimization for
   * finding the optimal value for the dampening factor \(\alpha\) such
   * that the resultant forecast is as close to the sample as possible.
   */
  public SingleExponentialSmoothingForecastModel()
  {
    this(AlphaOptimizer.LEAST_SQUARES);
  }

  /**
   * Creates a model with simple average strategy for generating the first
   * prediction and a specified strategy for finding the optimal value for
   * the dampening factor \(\alpha\).
   *
   * @param alphaOptimizer The strategy to use for optimizing \(\alpha\) such
   *                       that the forecast is as close to the sample as
   *                       possible.
   * @throws NullPointerException if {@code alphaOptimizer} is {@literal null}.
   * @see AlphaOptimizer
   */
  private SingleExponentialSmoothingForecastModel(final AlphaOptimizer alphaOptimizer)
  {
    this(FirstPredictionGenerator.SIMPLE_AVERAGE, alphaOptimizer);
  }

  /**
   * Creates a model with a specified strategy for generating the first
   * prediction.
   *
   * @param firstPredictionGenerator The strategy to use for generating the
   *                                 first prediction. This has a significant
   *                                 impact on the model accuracy, especially
   *                                 if a small \(\alpha\) is used. It is
   *                                 recommended to use the simple average of
   *                                 the observations as the initial
   *                                 prediction, which, even though slightly
   *                                 time-consuming, is likely to produce a
   *                                 more accurate forecast.
   * @param alphaOptimizer           The strategy to use for optimizing
   *                                 \(\alpha\) such that the forecast is as
   *                                 close to the sample as possible.
   * @throws NullPointerException if {@code firstPredictionGenerator} or
   *                              {@code alphaOptimizer} is {@literal null}.
   * @see AlphaOptimizer
   * @see FirstPredictionGenerator
   */
  private SingleExponentialSmoothingForecastModel(final FirstPredictionGenerator firstPredictionGenerator, final AlphaOptimizer alphaOptimizer)
  {
    super();

    // Ensure that the alpha optimization strategy is specified.
    if (alphaOptimizer == null)
    {
      throw new NullPointerException("The strategy for optimizing alpha must be specified.");
    }

    // Ensure that the first prediction generation strategy is specified.
    if (firstPredictionGenerator == null)
    {
      throw new NullPointerException("The strategy for generating the first prediction must be specified.");
    }

    this.alphaOptimizer = alphaOptimizer;
    this.firstPredictionGenerator = firstPredictionGenerator;
  }

  /**
   * Creates a baseline of a collection of observations. The baseline is
   * used to generate predictions for the observations, as well as fine-tune
   * the model to produce optimal predictions.
   *
   * @param observations A collection of observations.
   * @return A baseline version of the observations.
   */
  double[] baselineObservations(final double[] observations)
  {
    // In Hunter's model, each observation is the baseline for its next
    // observation, that is to say, the prediction for each observation
    // is directly dependent not upon that observation, but its immediately
    // preceding observation.
    final double[] baseline = new double[observations.length];

    System.arraycopy(observations, 0, baseline, 1, observations.length - 1);

    return baseline;
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

    // Smoothen the observations.
    final double[] smoothed = smoothenObservations(observations, optimalAlpha);

    // Add the smooth observations as the predictions for the known
    // observations.
    final double[] predictions = Arrays.copyOf(smoothed, observations.length + projections);

    // Add a prediction corresponding to the last observation.
    predictions[observations.length] = optimalAlpha * observations[observations.length - 1]
        + (1 - optimalAlpha) * predictions[observations.length - 1];

    // Add specified number of predictions beyond the sample.
    for (int j = 1; j < projections; ++j)
    {
      predictions[observations.length + j] = predictions[observations.length];
    }

    return forecast(observations, predictions);
  }

  /**
   * Gets the optimal value for the dampening factor \(\alpha\) for a set of
   * observations.
   *
   * @param observations The observations for which \(\alpha\) is required.
   * @return A best-effort optimal value for \(\alpha\), given the
   * observations and the initial guess.
   */
  double optimalAlpha(final double[] observations)
  {
    return alphaOptimizer.optimize(observations, this::baselineObservations, this::smoothenObservations);
  }

  /**
   * Smoothens a collection of observations by exponentially dampening them
   * using a factor \(\alpha\), using each observation to generate its next
   * (instead of corresponding) prediction.
   *
   * @param observations The observations to smoothen.
   * @param alpha        The dampening factor \(\alpha\).
   * @return Smoothened observations.
   */
  double[] smoothenObservations(final double[] observations, final double alpha)
  {
    final double[] smoothed = new double[observations.length];

    // Generate the first smooth observation using a specific strategy.
    smoothed[0] = firstPrediction(observations);

    // Prepare a baseline for the observations.
    final double[] baseline = baselineObservations(observations);

    // Generate the rest using the smoothing formula and using the baseline
    // values instead of the observations directly.
    for (int i = 1; i < observations.length; ++i)
    {
      smoothed[i] = smoothed[i - 1] + alpha * (baseline[i] - smoothed[i - 1]);
    }

    return smoothed;
  }

  /**
   * Gets the first prediction from a sample of observations, based on the
   * strategy specified for the model.
   *
   * @param observations The observations for which the first prediction is
   *                     required.
   * @return The first prediction for the observations based on the chosen
   * strategy.
   */
  private double firstPrediction(final double[] observations)
  {
    return firstPredictionGenerator.predict(observations);
  }

  /**
   * Determines how the dampening factor \(\alpha\) is optimized such that
   * the resultant forecast is as close to the sample as possible.
   */
  public enum AlphaOptimizer
  {
    /**
     * Optimizes the value for the dampening factor \(\alpha\) for a set of
     * observations using the {@literal Levenberg-Marquardt (LM)} algorithm.
     *
     * @see <a href="https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm">Levenberg-Marquardt algorithm</a>
     */
    LEAST_SQUARES
        {
          private final LevenbergMarquardtOptimizer ALPHA_OPTIMIZER = new LevenbergMarquardtOptimizer();

          /**
           * {@inheritDoc}
           */
          @Override
          protected double optimize(final double[] observations
              , final Function<double[], double[]> baselineFunction
              , final BiFunction<double[], Double, double[]> smoothingFunction)
          {
            return ALPHA_OPTIMIZER.optimize(optimizationModel(observations, baselineFunction, smoothingFunction))
                                  .getPoint()
                                  .getEntry(0);
          }

          /**
           * Converts a collection of observations into a
           * {@literal least-squares curve-fitting problem} using an initial
           * value of the dampening factor \(\alpha\). The dampening factor is
           * used for exponential smoothing so that the resultant problem can
           * be fed to a curve-fitting algorithm for finding the optimal value
           * of \(\alpha\).
           *
           * @param observations The observations to convert to a least-squares
           *                     curve-fitting problem.
           * @param baselineFunction  A function that prepares a baseline for
           *                          the observations. The baseline in turn is
           *                          used to generate predictions for the
           *                          observations and also to optimize the
           *                          value of \(\alpha\) through an iterative
           *                          process that attempts to solve the
           *                          least-squares problem.
           * @param smoothingFunction A function to use for smoothing the
           *                         observations during the optimization
           *                         process.
           * @return A {@link LeastSquaresProblem}.
           */
          private LeastSquaresProblem optimizationModel(final double[] observations
              , final Function<double[], double[]> baselineFunction
              , final BiFunction<double[], Double, double[]> smoothingFunction)
          {
            return new LeastSquaresBuilder()
                .checkerPair(ALPHA_CONVERGENCE_CHECKER)
                .maxEvaluations(MAX_ALPHA_OPTIMIZATION_EVALUATIONS)
                .maxIterations(MAX_ALPHA_OPTIMIZATION_ITERATIONS)
                .model(optimizationModelFunction(observations, smoothingFunction), optimizationModelJacobian(observations, baselineFunction, smoothingFunction))
                .parameterValidator(alphaValidator())
                .target(observations)
                .start(new double[] { INITIAL_ALPHA })
                .build();
          }

          /**
           * Gets the model function to use for optimizing the value of \(\alpha\),
           * which is basically just the predicted value for each observed value. The
           * optimization algorithm then uses the difference between the observed
           * and corresponding predicted values to find the optimal value for
           * \(\alpha\).
           *
           * @param observations The observations for which the optimal value of
           *                     \(\alpha\) is required.
           * @param smoothingFunction A function to use for smoothing the
           *                         observations during the optimization
           *                         process.
           * @return A {@link MultivariateVectorFunction}.
           */
          private MultivariateVectorFunction optimizationModelFunction(final double[] observations, final BiFunction<double[], Double, double[]> smoothingFunction)
          {
            return params -> smoothingFunction.apply(observations, params[0]);
          }

          /**
           * <p>
           * Gets the {@literal Jacobian} corresponding to the model function
           * used for optimizing the value of \(\alpha\).
           * </p>
           *
           * <p>
           * The {@literal Jacobian} (\(J\)) for a function \(S\) of \(k\)
           * parameters \(x_k\) is a matrix such that an element \(J_{ik}\) of
           * the matrix is given by
           * \(J_{ik} = \frac{\partial S_i}{\partial x_k}\), where \(x_k\)
           * are the \(k\) parameters on which the function \(S\) is dependent,
           * and \(S_i\) are the values of the function \(S\) at \(i\) distinct
           * points. For a function \(S\) that is differentiable with respect
           * to \(x_k\) at each point \(i\), the {@literal Jacobian} (\(J\)) is
           * a good linear approximation of the geometric shape of the function
           * \(S\) in the immediate vicinity of each \(x_k\). This allows it to
           * be used as a sort of derivative for the function \(S\), wherever a
           * derivative is required to determine the slope of the function at
           * any given point. The slope in turn determines whether the function
           * is ascending or descending at any given point, and helps the
           * optimization process choose a direction in which to change the
           * value of \(\alpha\) so that its optimal value can be found as
           * quickly as possible.
           * </p>
           *
           * <p>
           * In the case of single exponential smoothing forecast models, there
           * is only one parameter \(\alpha\), since the predicted values only
           * depend on this one parameter. Therefore, the {@literal Jacobian}
           * (\(J\)) depends on only this one parameter \(\alpha\). Therefore,
           * (\(J\)) reduces to \(J_{i} = \frac{\partial S_i}{\partial \alpha}\)
           * (\(k = 1\)).
           * </p>
           *
           * <p>
           * For the single exponential smoothing model, the function \(S\) is
           * defined as (see above)
           * </p>
           * <p>
           * <br>
           * \(S_i = S_{i-1} + \alphaD\)
           * <br>
           * </p>
           * <p>
           * where, \(D\) is dependent upon some observation and the previous
           * value \(S_{i-1}), therefore,
           * \(J_{i} = \frac{\partial S_i}{\partial \alpha}\) reduces to
           * \(J_{i} = \frac{\partial S_{i-1}}{\partial \alpha} + \alpha{\frac{\partial D}{\partial \alpha}} + D\frac{\partial \alpha}{\partial \alpha}\).
           * </p>
           * <p>
           * Since, \(S_{i-1}\) may be dependent upon \(\alpha\), this formula
           * needs to be evaluated for each \(S_i\) to calculate the correct
           * \(J_i\).
           * </p>
           *
           * @param observations The observations for which the optimal value of
           *                     \(\alpha\) is required.
           * @param baselineFunction  A function that prepares a baseline for
           *                          the observations. The baseline in turn is
           *                          used to generate predictions for the
           *                          observations and also to optimize the
           *                          value of \(\alpha\) through an iterative
           *                          process that attempts to solve the
           *                          least-squares problem.
           * @param smoothingFunction A function to use for smoothing the
           *                         observations during the optimization
           *                         process.
           * @return A {@link MultivariateMatrixFunction}.
           * @see <a href="https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant">Jacobian matrix</a>
           */
          private MultivariateMatrixFunction optimizationModelJacobian(final double[] observations
              , final Function<double[], double[]> baselineFunction
              , final BiFunction<double[], Double, double[]> smoothingFunction)
          {
            return params -> {
              final double alpha = params[0];

              // Smoothen the observations.
              final double[] smoothed = smoothingFunction.apply(observations, alpha);

              final double[][] jacobian = new double[observations.length][1];

              // Prepare a baseline for the observations.
              final double[] baseline = baselineFunction.apply(observations);

              // The first element of the Jacobian is simply the first observation,
              // since there is no prior prediction for it.
              jacobian[0][0] = observations[0];

              // Calculate the rest of the Jacobian using the current observation
              // and the immediately previous prediction.
              for (int i = 1; i < jacobian.length; ++i)
              {
                jacobian[i][0] = baseline[i] - smoothed[i - 1]
                    + Math.min(1, i - 1) * Math.pow(1 - alpha, i - 1) * jacobian[i - 1][0];
              }

              return jacobian;
            };
          }
        };

    protected final SimpleVectorValueChecker ALPHA_CONVERGENCE_CHECKER = new SimpleVectorValueChecker(1e-6, 1e-6);

    /**
     * Gets a validator that ensures that the value of \(\alpha\) as determined
     * by the optimization algorithm remains within its expected bounds during
     * all iterations.
     *
     * @return A {@link ParameterValidator}.
     */
    protected ParameterValidator alphaValidator()
    {
      return points -> {
        final double alpha = points.getEntry(0);

        return alpha >= MIN_ALPHA && alpha <= MAX_ALPHA
               // If the alpha is within its bounds, return its value.
               ? points
               : alpha < MIN_ALPHA
                 // If the alpha is below the lower threshold, reset it to the
                 // lower threshold.
                 ? new ArrayRealVector(new double[] { MIN_ALPHA })
                 // If the alpha is above the upper threshold, reset it to the
                 // upper threshold.
                 : new ArrayRealVector(new double[] { MAX_ALPHA });
      };
    }

    /**
     * Finds the optimal value for the dampening factor \(\alpha\) for a
     * collection of observations, such that the value can be used to
     * generate a forecast that is as close to the sample as possible.
     *
     * @param observations      The observations for which the optimal value of
     *                          \(\alpha\) is required.
     * @param baselineFunction  A function that prepares a baseline for the
     *                          observations. The baseline in turn is used to
     *                          generate predictions for the observations and
     *                          also to optimize the value of \(\alpha\).
     * @param smoothingFunction A smoothing function to use for finding smooth
     *                          versions of the observations during the
     *                          optimization process.
     * @return An optimal value for \(\alpha\) for the given observations.
     */
    protected abstract double optimize(final double[] observations
        , final Function<double[], double[]> baselineFunction
        , final BiFunction<double[], Double, double[]> smoothingFunction);
  }

  /**
   * Determines how the first prediction for a sample of observations should be
   * generated.
   */
  public enum FirstPredictionGenerator
  {
    /**
     * Uses the first observation as the first prediction. A simple and quick
     * strategy that may lead to large errors when used with a small
     * \(\alpha\).
     */
    FIRST_OBSERVATION
        {
          @Override
          double predict(final double[] observations)
          {
            return observations[0];
          }
        },

    /**
     * Uses the simple average of all the observations as the first prediction.
     */
    SIMPLE_AVERAGE
        {
          @Override
          double predict(final double[] observations)
          {
            return SingleExponentialSmoothingForecastModel.simpleAverage(observations);
          }
        };

    /**
     * Determines the prediction corresponding to the first observation
     * in a given collection of observations.
     *
     * @param observations The observations for which the prediction is
     *                     required.
     * @return The first prediction for the observed values.
     */
    abstract double predict(final double[] observations);
  }
}
