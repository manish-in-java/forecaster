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
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;

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
 * \(\boxed{S_i = \alpha{O_{i-1}} + (1 - \alpha)S_{i-1}}\)
 * <br>
 * </p>
 *
 * <p>or its alternative form</p>
 *
 * <p>
 * <br>
 * \(\boxed{S_i = S_{i-1} + \alpha({O_{i-1}} - S_{i-1})}\)
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
 * <br>
 * ... and so on.
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
public class SingleExponentialSmoothingForecastModel extends ExponentialSmoothingForecastModel
{
  private final AlphaOptimizer          alphaOptimizer;
  private final FirstPredictionProvider firstPredictionProvider;

  /**
   * Creates a model with simple average strategy for generating the first
   * prediction and a {@literal non-linear least-squares} optimizer for
   * finding the optimal value for the dampening factor \(\alpha\).
   */
  public SingleExponentialSmoothingForecastModel()
  {
    this(LeastSquaresOptimizer.INSTANCE);
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
  SingleExponentialSmoothingForecastModel(final AlphaOptimizer alphaOptimizer)
  {
    this(SimpleAverageFirstPredictionProvider.INSTANCE, alphaOptimizer);
  }

  /**
   * Creates a model with a specified strategy for generating the first
   * prediction.
   *
   * @param firstPredictionProvider The strategy to use for generating the
   *                                first prediction. This has a significant
   *                                impact on the model accuracy, especially
   *                                if a small \(\alpha\) is used. It is
   *                                recommended to use the simple average of
   *                                the observations as the initial
   *                                prediction, which, even though slightly
   *                                time-consuming, is likely to produce a
   *                                more accurate forecast.
   * @param alphaOptimizer          The strategy to use for optimizing
   *                                \(\alpha\) such that the forecast is as
   *                                close to the sample as possible.
   * @throws NullPointerException if {@code firstPredictionProvider} or
   *                              {@code alphaOptimizer} is {@literal null}.
   * @see AlphaOptimizer
   * @see FirstPredictionProvider
   */
  SingleExponentialSmoothingForecastModel(final FirstPredictionProvider firstPredictionProvider, final AlphaOptimizer alphaOptimizer)
  {
    super();

    // Ensure that the alpha optimization strategy is specified.
    if (alphaOptimizer == null)
    {
      throw new NullPointerException("The strategy for optimizing alpha must be specified.");
    }

    // Ensure that the first prediction generation strategy is specified.
    if (firstPredictionProvider == null)
    {
      throw new NullPointerException("The strategy for generating the first prediction must be specified.");
    }

    this.alphaOptimizer = alphaOptimizer;
    this.firstPredictionProvider = firstPredictionProvider;
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
    // Using an initial guess for alpha, find its optimal value that will
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
   * observations.
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
    return firstPredictionProvider.predict(observations);
  }

  /**
   * <p>
   * Determines how the dampening factor \(\alpha\) is optimized such that
   * the resultant forecast is as close to the sample as possible.
   * </p>
   *
   * <ul>
   * <li>For the general case, and when the predicted value \(S_i\) does
   * not depend upon the observed value \(O_i\) directly (that is, it only
   * depends upon observed values \(O_{i-1}\) or earlier), use
   * {@link LeastSquaresOptimizer} as it finds the optimal \(\alpha\) more
   * accurately and much faster than any other method. When the predicted
   * value \(S_i\) directly depends upon the observed value \(O_i\), the error,
   * which is calculated as \(O_i - S_i\) is zero when \(S_i = O_i\). This
   * tendency automatically pushes the optimizer towards \(\alpha = 1.0\),
   * when \(S_i = O_i\). This is why, {@link LeastSquaresOptimizer} should not
   * be used when \(S_i\) directly depends upon \(O_i\);</li>
   * <li>When the predicted value \(S_i\) directly depends upon the observed
   * value \(O_i\), use the {@link GradientDescentOptimizer}.</li>
   * </ul>
   */
  public abstract static class AlphaOptimizer
  {
    /**
     * Gets a validator that ensures that the value of \(\alpha\) as determined
     * by the optimization algorithm remains within its expected bounds during
     * all iterations.
     *
     * @return A {@link ParameterValidator}.
     */
    protected final ParameterValidator alphaValidator()
    {
      return points -> {
        final double alpha = points.getEntry(0);

        return alpha < MIN_DAMPENING_FACTOR
               // If alpha is below the lower threshold, reset it to the
               // lower threshold.
               ? new ArrayRealVector(new double[] { MIN_DAMPENING_FACTOR })
               : alpha > MAX_DAMPENING_FACTOR
                 // If alpha is above the upper threshold, reset it to the
                 // upper threshold.
                 ? new ArrayRealVector(new double[] { MAX_DAMPENING_FACTOR })
                 // If alpha is within its bounds, return its value.
                 : points;
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
   * <p>
   * Optimizes the value for the dampening factor \(\alpha\) for a set of
   * observations using the {@literal Non-linear Conjugate Gradient Descent}
   * method, which is a non-linear steepest-descent optimization algorithm.
   * This method generates predictions for the given observations, starting
   * with an initial guess for the exponent \(\alpha\), and iteratively
   * changing the dampening factor value until one is found where the gradient
   * of the curve generated by the predicted values becomes {@literal zero}.
   * </p>
   *
   * <p>
   * The optimization algorithm requires the following input:
   * </p>
   *
   * <ol>
   * <li>The observations for which \(\alpha\) needs to be optimized;</li>
   * <li>An initial guess for \(\alpha\);</li>
   * <li>A function that can take the observations and some value for
   * \(\alpha\), and produce predictions for each of the observations.
   * This is known as the model function for the algorithm. The algorithm
   * internally calculates the MSE for each value of \(\alpha\) by using
   * the given observations and their corresponding predictions for the
   * specified value of \(\alpha\);</li>
   * <li>A function that can take the observations and some value for
   * \(\alpha\), and produce a <i>gradient</i> for each prediction
   * made by the model function. The gradient determines whether the
   * value of \(\alpha\) is too high or too low, and therefore whether
   * it needs to be lowered or raised.</li>
   * </ol>
   *
   * @see <a href="https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method">Non-linear conjugate gradient descent</a>
   */
  public final static class GradientDescentOptimizer extends AlphaOptimizer
  {
    public static final AlphaOptimizer INSTANCE = new GradientDescentOptimizer();

    private final SimpleValueChecker                  ALPHA_CONVERGENCE_CHECKER = new SimpleValueChecker(DAMPENING_FACTOR_CONVERGENCE_THRESHOLD,
                                                                                                         DAMPENING_FACTOR_CONVERGENCE_THRESHOLD);
    private final NonLinearConjugateGradientOptimizer ALPHA_OPTIMIZER           = new NonLinearConjugateGradientOptimizer(
        NonLinearConjugateGradientOptimizer.Formula.FLETCHER_REEVES, ALPHA_CONVERGENCE_CHECKER);
    private final int                                 MAX_ITERATIONS            = 10000;

    /**
     * Deliberately hidden to prevent instantiation.
     */
    private GradientDescentOptimizer()
    {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected double optimize(final double[] observations
        , final Function<double[]
        , double[]> baselineFunction
        , final BiFunction<double[], Double, double[]> smoothingFunction)
    {
      return ALPHA_OPTIMIZER.optimize(optimizationModelFunction(observations, baselineFunction, smoothingFunction)
          , optimizationFunctionGradient(observations, baselineFunction, smoothingFunction)
          , GoalType.MINIMIZE
          , new InitialGuess(new double[] { INITIAL_DAMPENING_FACTOR })
          , new MaxEval(MAX_ITERATIONS)
          , new MaxIter(MAX_ITERATIONS))
                            .getPoint()[0];
    }

    /**
     * Gets the model function to use for finding the gradient for a
     * given value of alpha.
     *
     * @param observations      The observations for which the optimal value of
     *                          \(\alpha\) is required.
     * @param baselineFunction  A function that prepares a baseline for
     *                          the observations. The baseline in turn is
     *                          used to generate predictions for the
     *                          observations and also to optimize the
     *                          value of \(\alpha\) through an iterative
     *                          process.
     * @param smoothingFunction A function to use for smoothing the
     *                          observations during the optimization
     *                          process.
     * @return An {@link ObjectiveFunctionGradient}.
     */
    private ObjectiveFunctionGradient optimizationFunctionGradient(final double[] observations
        , final Function<double[], double[]> baselineFunction
        , final BiFunction<double[], Double, double[]> smoothingFunction)
    {
      return new ObjectiveFunctionGradient(params ->
                                           {
                                             final double alpha = params[0];

                                             // Prepare a baseline for the observations.
                                             final double[] baseline = baselineFunction.apply(observations);

                                             // Smoothen the baselined observations.
                                             final double[] smoothed = smoothingFunction.apply(baseline, alpha);

                                             double previousPrediction = smoothed[0];
                                             double previousSlope = 0.0;
                                             double gradient = 0.0;

                                             // Calculate the gradient.
                                             for (int i = 0; i < baseline.length - 1; ++i)
                                             {
                                               final double error = baseline[i + 1] - smoothed[i];

                                               final double slope = baseline[i] - previousPrediction + (1 - alpha) * previousSlope;

                                               gradient += error * slope;
                                               previousSlope = slope;
                                               previousPrediction = smoothed[i];
                                             }

                                             return new double[] { 2 * gradient };
                                           });
    }

    /**
     * Gets the model function to use for optimizing the value of
     * \(\alpha\), which calculates the MSE for a given collection of
     * observations and a specified value of alpha\(\alpha\).
     *
     * @param observations      The observations for which the optimal value of
     *                          \(\alpha\) is required.
     * @param baselineFunction  A function that prepares a baseline for
     *                          the observations. The baseline in turn is
     *                          used to generate predictions for the
     *                          observations and also to optimize the
     *                          value of \(\alpha\) through an iterative
     *                          process.
     * @param smoothingFunction A function to use for smoothing the
     *                          observations during the optimization
     *                          process.
     * @return An {@link ObjectiveFunction}.
     */
    private ObjectiveFunction optimizationModelFunction(final double[] observations
        , final Function<double[], double[]> baselineFunction
        , final BiFunction<double[], Double, double[]> smoothingFunction)
    {
      return new ObjectiveFunction(params -> {
        final double alpha = params[0];

        // Prepare a baseline for the observations.
        final double[] baseline = baselineFunction.apply(observations);

        // Smoothen the baselined observations.
        final double[] smoothed = smoothingFunction.apply(baseline, alpha);

        double sse = 0.0;
        for (int i = 0; i < baseline.length - 1; ++i)
        {
          final double error = baseline[i + 1] - smoothed[i];

          sse += error * error;
        }

        return sse;
      });
    }
  }

  /**
   * <p>
   * Optimizes the value for the dampening factor \(\alpha\) for a set of
   * observations using the {@literal Levenberg-Marquardt (LM)} method,
   * which is a non-linear least-squares curve-fitting algorithm. This
   * method generates predictions for the given observations, starting
   * with an initial guess for the dampening factor \(\alpha\), and iteratively
   * changing the dampening factor value until one is found that minimizes the
   * {@literal sum-squared-error (MSE)} for the observations.
   * </p>
   *
   * <p>
   * The optimization algorithm requires the following input:
   * </p>
   *
   * <ol>
   * <li>The observations for which \(\alpha\) needs to be optimized;</li>
   * <li>An initial guess for \(\alpha\);</li>
   * <li>A function that can take the observations and some value for
   * \(\alpha\), and produce predictions for each of the observations.
   * This is known as the model function for the algorithm. The algorithm
   * internally calculates the MSE for each value of \(\alpha\) by using
   * the given observations and their corresponding predictions for the
   * specified value of \(\alpha\);</li>
   * <li>A function that can take the observations and some value for
   * \(\alpha\), and produce a <i>Jacobian</i> for each prediction
   * made by the model function. The <i>Jacobian</i> is a linear
   * approximation of the derivative of a predicted value with respect
   * to the variable parameter \(\alpha\) and hence serves to determine
   * whether the trend at each predicted value matches that of the
   * corresponding observed value. This information is critical in optimizing
   * the value of \(\alpha\) as the derivative determines whether the
   * value of \(\alpha\) is too high or too low, and therefore whether
   * it needs to be lowered or raised;</li>
   * <li>A function that can validate whether a specific value of
   * \(\alpha\) is within the bounds of the problem-space. For the
   * purposes of exponential smoothing, \(\alpha\) must be between
   * {@literal 0.1} and {@literal 0.9}.</li>
   * </ol>
   *
   * @see <a href="https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm">Levenberg-Marquardt algorithm</a>
   * @see <a href="https://en.wikipedia.org/wiki/Least_squares">Least squares</a>
   * @see <a href="https://en.wikipedia.org/wiki/Curve_fitting">Curve fitting</a>
   */
  public final static class LeastSquaresOptimizer extends AlphaOptimizer
  {
    public static final AlphaOptimizer INSTANCE = new LeastSquaresOptimizer();

    /**
     * Deliberately hidden to prevent instantiation.
     */
    private LeastSquaresOptimizer()
    {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected double optimize(final double[] observations
        , final Function<double[], double[]> baselineFunction
        , final BiFunction<double[], Double, double[]> smoothingFunction)
    {
      return LEAST_SQUARES_OPTIMIZER.optimize(optimizationModel(observations, baselineFunction, smoothingFunction))
                                    .getPoint()
                                    .getEntry(0);
    }

    /**
     * Converts a collection of observations into a
     * {@literal least-squares curve-fitting problem} using an initial
     * value of the dampening factor \(\alpha\). The dampening factor is used for
     * exponential smoothing so that the resultant problem can be fed
     * to a curve-fitting algorithm for finding the optimal value of
     * \(\alpha\).
     *
     * @param observations      The observations to convert to a least-squares
     *                          curve-fitting problem.
     * @param baselineFunction  A function that prepares a baseline for
     *                          the observations. The baseline in turn is
     *                          used to generate predictions for the
     *                          observations and also to optimize the
     *                          value of \(\alpha\) through an iterative
     *                          process that attempts to solve the
     *                          least-squares problem.
     * @param smoothingFunction A function to use for smoothing the
     *                          observations during the optimization
     *                          process.
     * @return A {@link LeastSquaresProblem}.
     */
    private LeastSquaresProblem optimizationModel(final double[] observations
        , final Function<double[], double[]> baselineFunction
        , final BiFunction<double[], Double, double[]> smoothingFunction)
    {
      return new LeastSquaresBuilder()
          .checkerPair(DAMPENING_FACTOR_CONVERGENCE_CHECKER)
          .maxEvaluations(MAX_OPTIMIZATION_EVALUATIONS)
          .maxIterations(MAX_OPTIMIZATION_ITERATIONS)
          .model(optimizationModelFunction(observations, smoothingFunction), optimizationModelJacobian(observations, baselineFunction, smoothingFunction))
          .parameterValidator(alphaValidator())
          .target(observations)
          .start(new double[] { INITIAL_DAMPENING_FACTOR })
          .build();
    }

    /**
     * Gets the model function to use for optimizing the value of \(\alpha\),
     * which is just the predicted value for each observed value. The
     * optimization algorithm then uses the difference between the observed
     * and corresponding predicted values to find the optimal value for
     * \(\alpha\).
     *
     * @param observations      The observations for which the optimal value of
     *                          \(\alpha\) is required.
     * @param smoothingFunction A function to use for smoothing the
     *                          observations during the optimization
     *                          process.
     * @return A {@link MultivariateVectorFunction}.
     */
    private MultivariateVectorFunction optimizationModelFunction(final double[] observations, final BiFunction<double[], Double, double[]> smoothingFunction)
    {
      return params -> smoothingFunction.apply(observations, params[0]);
    }

    /**
     * <p>
     * Gets the <i>Jacobian</i> (\(J\)) corresponding to the model function
     * used for optimizing the value of \(\alpha\). The <i>Jacobian</i> for a
     * function \(S\) of \(k\) parameters \(x_k\) is a matrix, where an element
     * \(J_{ik}\) of the matrix is given by
     * \(J_{ik} = \frac{\partial S_i}{\partial x_k}\), where, \(x_k\) are
     * the \(k\) parameters on which the function \(S\) is dependent, and
     * \(S_i\) are the values of the function \(S\) at \(i\) distinct points.
     * In the case of single exponential smoothing forecast models, there
     * is only one parameter \(\alpha\) that impacts the predicted value for
     * a given observed value. Therefore, the <i>Jacobian</i> (\(J\)) depends
     * on only this one parameter \(\alpha\) and thereby reduces to
     * \(\boxed{J_i = \frac{\partial S_i}{\partial \alpha}}\).
     * </p>
     *
     * <p>
     * For the single exponential smoothing model, the function \(S\) is
     * defined as (see above)
     * </p>
     *
     * <p>
     * <br>
     * \(S_i = S_{i-1} + \alpha(B_i - S_{i-1})\)
     * <br>
     * </p>
     *
     * <p>
     * where, \(B_i\) is a representation of the observation \(O_i\), referred
     * to here as the baseline version of that observation. Therefore,
     * </p>
     *
     * <p>
     * <br>
     * \(J_i = \frac{\partial S_i}{\partial \alpha}\)
     * <br>
     * becomes
     * <br><br>
     * \(J_i = \frac{\partial (S_{i-1} + \alpha(B_i - S_{i-1}))}{\partial \alpha}\),
     * <br>
     * or (by the associative rule of differentiation)
     * <br><br>
     * \(J_i = \frac{\partial S_{i-1}}{\partial \alpha} + \frac{\partial (\alpha(B_i - S_{i-1}))}{\partial \alpha}\),
     * <br>
     * or (by the associative rule of differentiation)
     * <br><br>
     * \(J_i = \frac{\partial S_{i-1}}{\partial \alpha} + \frac{\partial (\alpha{B_i})}{\partial \alpha} - \frac{\partial (\alpha{S_{i-1}})}{\partial \alpha}\),
     * <br>
     * or (by the chain rule of differentiation)
     * <br><br>
     * \(J_i = \frac{\partial S_{i-1}}{\partial \alpha}
     * + \alpha{\frac{\partial B_i}{\partial \alpha}}
     * + B_i{\frac{\partial \alpha}{\partial \alpha}}
     * - \alpha{\frac{\partial S_{i-1}}{\partial \alpha}}
     * - S_{i-1}{\frac{\partial \alpha}{\partial \alpha}}\),
     * <br>
     * or (upon simplification)
     * <br><br>
     * \(J_i = \frac{\partial S_{i-1}}{\partial \alpha}
     * + B_i
     * - \alpha{\frac{\partial S_{i-1}}{\partial \alpha}}
     * - S_{i-1}\)
     * <br>
     * (since \(\frac{\partial B_i}{\partial \alpha} = 0\), given that
     * \(B_i\), which is only dependent upon the observations does not
     * depend on \(\alpha\)), or (upon rearrangement of terms)
     * <br><br>
     * \(J_i = B_i - S_{i-1} + (1 - \alpha)\frac{\partial S_{i-1}}{\partial \alpha}\),
     * <br>
     * or, using the fact that \(\frac{\partial S_{i-1}}{\partial \alpha} = J_{i-1}\)
     * <br><br>
     * \(\boxed{J_i = B_i - S_{i-1} + (1 - \alpha)J_{i-1}}\)
     * </p>
     *
     * <p>
     * Since \(S_1\) is not dependent upon \(\alpha\),
     * \(J_1 = \frac{\partial S_1}{\partial \alpha} = 0\). This gives an
     * initial value for the Jacobian that can be used to derive the rest
     * using the recursive formula derived above.
     * </p>
     *
     * @param observations      The observations (\(O_i\)) for which the
     *                          optimal value of \(\alpha\) is required.
     * @param baselineFunction  A function that prepares a baseline \(B_i\)
     *                          for each observation \(O_i\). The baseline
     *                          in turn is used to generate predictions for
     *                          the observations and also to optimize the
     *                          value of \(\alpha\) through an iterative
     *                          process that attempts to solve the
     *                          least-squares problem.
     * @param smoothingFunction A function to use for smoothing the
     *                          observations during the optimization
     *                          process.
     * @return A {@link MultivariateMatrixFunction}, which is a
     * single-column matrix whose elements correspond to the
     * <i>Jacobian</i> of the model function.
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
  }

  /**
   * Provides the first prediction for a sample of observations.
   */
  public abstract static class FirstPredictionProvider
  {
    /**
     * Provides the prediction corresponding to the first observation
     * in a given collection of observations.
     *
     * @param observations The observations for which the prediction is
     *                     required.
     * @return The first prediction for the observed values.
     */
    protected abstract double predict(final double[] observations);
  }

  /**
   * Uses the simple average of all the observations as the first prediction.
   */
  public final static class SimpleAverageFirstPredictionProvider extends FirstPredictionProvider
  {
    public static final FirstPredictionProvider INSTANCE = new SimpleAverageFirstPredictionProvider();

    /**
     * Deliberately hidden to prevent instantiation.
     */
    private SimpleAverageFirstPredictionProvider()
    {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected double predict(final double[] observations)
    {
      return SingleExponentialSmoothingForecastModel.simpleAverage(observations);
    }
  }

  /**
   * Uses the first observation as the first prediction. A simple and quick
   * strategy that may lead to large errors when used with a small value for
   * \(\alpha\).
   */
  public final static class SimpleFirstPredictionProvider extends FirstPredictionProvider
  {
    public static final FirstPredictionProvider INSTANCE = new SimpleFirstPredictionProvider();

    /**
     * Deliberately hidden to prevent instantiation.
     */
    private SimpleFirstPredictionProvider()
    {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected double predict(final double[] observations)
    {
      return observations[0];
    }
  }
}
