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
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient;
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * <p>
 * Generates forecast for a sample using exponential smoothing, where an
 * observation \(\bf y\) is dampened exponentially to get a smooth version
 * \(\bf l\) as
 * </p>
 *
 * <p>
 * <br>
 * \(\large \boxed{l_t = \alpha{y_{t-1}} + (1 - \alpha)l_{t-1}}\)
 * <br>
 * </p>
 *
 * <p>or its alternative form</p>
 *
 * <p>
 * <br>
 * \(\large \boxed{l_t = l_{t-1} + \alpha({y_{t-1}} - l_{t-1})}\)
 * <br>
 * </p>
 *
 * <p>
 * where, \(\bf t\) is an index that ranges from {@literal 1} to the number of
 * observations in the sample, \(\bf y_t\) is the {@literal t-th} observation,
 * \(\bf l_t\) its smooth version, and \(\bf \alpha\) is a dampening factor
 * between \(\bf 0.0\) and \(\bf 1.0\) responsible for smoothing out the
 * observations. This means
 * </p>
 *
 * <p>
 * \(\large l_2 = \alpha{y_1} + (1 - \alpha)l_1\)
 * <br>
 * \(\large l_3 = \alpha{y_2} + (1 - \alpha)l_2\)
 * <br>
 * \(\large l_4 = \alpha{y_3} + (1 - \alpha)l_3\)
 * <br>
 * ... and so on.
 * </p>
 *
 * <p>
 * \(l_1\) can be chosen using one of many possible strategies, one of which
 * must be provided at the time of initializing the model.
 * </p>
 *
 * <p>
 * This model, characterized by a dependence of \(l_t\) upon \(y_{t-1}\) was
 * originally proposed in <i>1986</i> by <i>J. Stuart Hunter</i>.
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
  private final NonLinearConjugateGradientOptimizer OPTIMIZER = new NonLinearConjugateGradientOptimizer(NonLinearConjugateGradientOptimizer.Formula.FLETCHER_REEVES
      , new SimpleValueChecker(DAMPENING_FACTOR_CONVERGENCE_THRESHOLD, DAMPENING_FACTOR_CONVERGENCE_THRESHOLD));

  private final FirstPredictionProvider firstPredictionProvider;

  /**
   * Creates a model with simple average strategy for generating the first
   * prediction.
   */
  public SingleExponentialSmoothingForecastModel()
  {
    this(SimpleAverageFirstPredictionProvider.INSTANCE);
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
   * @throws NullPointerException if {@code firstPredictionProvider} is
   *                              {@literal null}.
   * @see FirstPredictionProvider
   */
  SingleExponentialSmoothingForecastModel(final FirstPredictionProvider firstPredictionProvider)
  {
    super();

    // Ensure that the first prediction generation strategy is specified.
    if (firstPredictionProvider == null)
    {
      throw new NullPointerException("The strategy for generating the first prediction must be specified.");
    }

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
  final double optimalAlpha(final double[] observations)
  {
    return optimize(observations, this::baselineObservations, this::smoothenObservations);
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
  private double optimize(final double[] observations
      , final Function<double[], double[]> baselineFunction
      , final BiFunction<double[], Double, double[]> smoothingFunction)
  {
    return OPTIMIZER.optimize(optimizationModelFunction(observations, baselineFunction, smoothingFunction)
        , optimizationFunctionGradient(observations, baselineFunction, smoothingFunction)
        , OPTIMIZATION_GOAL
        , new InitialGuess(new double[] { INITIAL_DAMPENING_FACTOR })
        , new MaxEval(MAX_OPTIMIZATION_EVALUATIONS)
        , new MaxIter(MAX_OPTIMIZATION_ITERATIONS))
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
