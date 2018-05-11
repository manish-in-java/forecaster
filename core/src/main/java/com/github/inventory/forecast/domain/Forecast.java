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

import com.github.inventory.forecast.model.ForecastModel;

import java.util.Collection;
import java.util.List;
import java.util.Vector;

/**
 * <p>
 * Represents a collection of predictions. A forecast is generated on a
 * {@link Sample} of {@link Observation}s by applying a
 * {@link ForecastModel}.
 * </p>
 * <p>
 * The applied {@link ForecastModel} generates
 * a prediction for every single {@link Observation} in the {@link Sample}.
 * Since {@link Observation}s are data points collected in reality, the
 * prediction generated for an {@link Observation} allows comparing the
 * actual value versus the predicted value.
 * </p>
 * <p>
 * In addition to predictions for each {@link Observation}, forecast models
 * also generate a specified number of additional predictions.
 * </p>
 * <p>
 * This class is thread-safe.
 * </p>
 */
public class Forecast
    extends Vector<Prediction>
    implements List<Prediction>, Collection<Prediction>
{
  /**
   * Constructs an empty forecast with capacity to hold the specified number of
   * predictions.
   *
   * @param size The number of predictions to include in the forecast beyond
   *             the sample size.
   * @throws IllegalArgumentException if the number of predictions is negative.
   */
  public Forecast(final int size)
  {
    super(size);
  }
}
