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

package com.github.inventory.forecast.domain;

import java.io.Serializable;

/**
 * <p>
 * Represents a prediction. A prediction can be {@literal defined} or
 * {@literal undefined}. When {@literal defined}, a prediction has a
 * {@literal value} and a {@literal mean-squared-error (MSE)}, which is a
 * measure of the prediction accuracy. For predictions obtained using
 * different forecast models from the same data, lower {@literal MSE}
 * signifies higher accuracy, and vice-versa.
 * </p>
 * <p>An {@literal undefined} prediction is one which cannot be determined
 * for the observed data by a particular forecast model. For example, if
 * a forecast model that requires one full year of past data to make
 * a prediction is invoked to generate a prediction with only 6 months of
 * data, the model may return an {@literal undefined} prediction to indicate
 * that it is unable to make the requested prediction.</p>
 */
public class Prediction implements Serializable
{
  private final double  bias;
  private final boolean defined;
  private final double  meanAbsoluteDeviation;
  private final double  meanAbsolutePercentageError;
  private final double  meanSquaredError;
  private final String  textRepresentation;
  private final double  totalAbsoluteError;
  private final double  value;

  /**
   * Creates a prediction with a given value and {@literal mean-squared-error}.
   *
   * @param value                       The predicted value.
   * @param totalAbsoluteError          The {@literal total-absolute-error}
   *                                    ({@literal SAE}) for the prediction,
   *                                    which is the sum of the absolute values
   *                                    of the differences between the
   *                                    predicted and actual values.
   * @param bias                        The bias, or average error for the
   *                                    prediction.
   * @param meanSquaredError            The {@literal mean-squared-error}
   *                                    ({@literal MSE}) for the prediction,
   *                                    which is the average of the squares of
   *                                    the differences between the predicted
   *                                    and actual values.
   * @param meanAbsoluteDeviation       The {@literal mean-absolute-deviation}
   *                                    ({@literal MAD}) for the prediction,
   *                                    which is the average of the sum of the
   *                                    absolute values of the differences
   *                                    between the predicted and actual
   *                                    values.
   * @param meanAbsolutePercentageError The
   *                                    {@literal mean-absolute-percentage-error}
   *                                    ({@literal MAPE}) for the prediction,
   *                                    which is the average of the average
   *                                    values of the differences between the
   *                                    predicted and actual values, divided by
   *                                    the actual values.
   */
  public Prediction(final double value
      , final double totalAbsoluteError
      , final double bias
      , final double meanSquaredError
      , final double meanAbsoluteDeviation
      , final double meanAbsolutePercentageError)
  {
    this(true, value, totalAbsoluteError, bias, meanSquaredError, meanAbsoluteDeviation, meanAbsolutePercentageError);
  }

  /**
   * Creates an {@literal undefined} prediction, which does not have a value.
   */
  private Prediction()
  {
    this(false, 0, 0, 0, 0, 0, 0);
  }

  /**
   * Creates a prediction with a given value and {@literal mean-squared-error}.
   *
   * @param defined                     Whether the prediction hs a defined value.
   * @param value                       The predicted value.
   * @param totalAbsoluteError          The {@literal total-absolute-error}
   *                                    ({@literal SAE}) for the prediction,
   *                                    which is the sum of the absolute values
   *                                    of the differences between the
   *                                    predicted and actual values.
   * @param bias                        The bias, or average error for the
   *                                    prediction.
   * @param meanSquaredError            The {@literal mean-squared-error}
   *                                    ({@literal MSE}) for the prediction,
   *                                    which is the average of the squares of
   *                                    the differences between the predicted
   *                                    and actual values.
   * @param meanAbsoluteDeviation       The {@literal mean-absolute-deviation}
   *                                    ({@literal MAD}) for the prediction,
   *                                    which is the average of the sum of the
   *                                    absolute values of the differences
   *                                    between the predicted and actual
   *                                    values.
   * @param meanAbsolutePercentageError The
   *                                    {@literal mean-absolute-percentage-error}
   *                                    ({@literal MAPE}) for the prediction,
   *                                    which is the average of the differences
   *                                    between the predicted and actual values,
   *                                    divided by the actual values.
   */
  private Prediction(final boolean defined
      , final double value
      , final double totalAbsoluteError
      , final double bias
      , final double meanSquaredError
      , final double meanAbsoluteDeviation
      , final double meanAbsolutePercentageError)
  {
    this.bias = bias;
    this.defined = defined;
    this.meanAbsoluteDeviation = meanAbsoluteDeviation;
    this.meanAbsolutePercentageError = meanAbsolutePercentageError;
    this.meanSquaredError = meanSquaredError;
    this.totalAbsoluteError = totalAbsoluteError;
    this.value = value;

    this.textRepresentation = getTextRepresentation();
  }

  /**
   * Creates an {@literal undefined} prediction, that is, one with no predicted
   * value.
   *
   * @return A {@link Prediction}.
   */
  public static Prediction undefined()
  {
    return new Prediction();
  }

  /**
   * Gets the bias for the prediction, which is the average error. For example,
   * if the sample is {@literal [10, 11, 12]} and the prediction is
   * {@literal 11}, the individual errors are
   * {@literal 11 - 10, 11 - 11, 11 - 12 = 1, 0, -1} and the bias
   * {@literal (1 + 0 - 1) / 3 = 0 / 3 = 0}.
   *
   * @return The bias for the prediction.
   */
  public double getBias()
  {
    return bias;
  }

  /**
   * Gets the {@literal mean-absolute-deviation} ({@literal MAD}) for the
   * prediction. For example, if the sample is {@literal [10, 11, 12]}
   * and the prediction is {@literal 11}, the individual errors are
   * {@literal 11 - 10, 11 - 11, 11 - 12 = 1, 0, -1}, absolute errors
   * are {@literal |1| = 1, |0| = 0, |-1| = 1} and
   * {@literal MAD} is {@literal (1 + 0 + 1) / 3 = 2 / 3 = 0.67}.
   *
   * @return The {@literal mean-absolute-deviation} for the prediction.
   * @see <a href="https://en.wikipedia.org/wiki/Mean_absolute_deviation">Mean Absolute Deviation</a>
   */
  public double getMeanAbsoluteDeviation()
  {
    return meanAbsoluteDeviation;
  }

  /**
   * Gets the {@literal mean-absolute-percentage-error} ({@literal MAPE})
   * for the prediction, which is the average of the differences between
   * the predicted and actual values, divided by the actual values.
   *
   * @return The {@literal mean-absolute-percentage-error} for the prediction.
   * @see <a href="https://en.wikipedia.org/wiki/Mean_absolute_percentage_error">Mean Absolute Percentage Error</a>
   */
  public double getMeanAbsolutePercentageError()
  {
    return meanAbsolutePercentageError;
  }

  /**
   * Gets the {@literal mean-squared-error} ({@literal MSE}) for the
   * prediction. For example, if the sample is {@literal [10, 11, 12]}
   * and the prediction is {@literal 11}, the individual errors are
   * {@literal 11 - 10, 11 - 11, 11 - 12 = 1, 0, -1}, squared errors
   * are {@literal 1 x 1 = 1, 0 x 0 = 0, -1 x -1 = 1} and
   * {@literal MSE} is {@literal (1 + 0 + 1) / 3 = 2 / 3 = 0.67}.
   *
   * @return The {@literal mean-squared-error} for the prediction.
   * @see <a href="https://en.wikipedia.org/wiki/Mean_squared_error">Mean Squared Error</a>
   */
  public double getMeanSquaredError()
  {
    return meanSquaredError;
  }

  /**
   * Gets the {@literal total-absolute-error} ({@literal SAE}) for the
   * prediction, which is the sum of the absolute values of the differences
   * between the predicted and actual values.
   *
   * @return The {@literal total-absolute-error} for the prediction.
   */
  public double getTotalAbsoluteError()
  {
    return totalAbsoluteError;
  }

  /**
   * Gets the predicted value.
   *
   * @return The predicted value.
   */
  public double getValue()
  {
    return value;
  }

  /**
   * Gets whether this prediction has a {@literal defined} value. Consumers
   * must check whether a prediction is {@literal defined}, before using it.
   *
   * @return Whether this prediction has a {@literal defined} value.
   */
  public boolean isDefined()
  {
    return defined;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public String toString()
  {
    return textRepresentation;
  }

  /**
   * Gets a printable text representation for this prediction.
   *
   * @return A printable text representation for this prediction.
   */
  private String getTextRepresentation()
  {
    return isDefined()
           ? String.format("Prediction(value : %f"
                               + ", SAE : %f"
                               + ", Bias : %f"
                               + ", MSE : %f"
                               + ", MAD : %f"
                               + ", MAPE : %f)"
        , getValue()
        , getTotalAbsoluteError()
        , getBias()
        , getMeanSquaredError()
        , getMeanAbsoluteDeviation()
        , getMeanAbsolutePercentageError())
           : "Prediction()";
  }
}
