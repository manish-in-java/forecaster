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
import com.github.forecast.domain.Sample;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Unit tests for {@link ExponentialSmoothingForecastModel}.
 */
abstract class ExponentialSmoothingForecastModelTest extends ForecastModelTest
{
  /**
   * Tests that an exponential smoothing forecast can be generated
   * correctly for a sample.
   */
  @Test
  public void testForecast()
  {
    testForecast(getForecastModel());
  }

  /**
   * Tests that an exponential smoothing forecast can be generated correctly
   * for a sample.
   *
   * @param model The model to use for generating the forecast.
   */
  final void testForecast(final ForecastModel model)
  {
    // Generate a random sample of observations.
    final Sample sample = getSample();

    // Generate a forecast for the sample.
    final Forecast subject = model.forecast(sample, getProjectionCount());
    final double[] predictions = subject.getPredictions();

    assertNotNull(subject);
    assertNotNull(predictions);
    assertTrue(predictions.length > sample.size());

    // Ensure that the accuracy measures have been calculated.
    assertNotEquals(0.0, subject.getBias());
    assertNotEquals(0.0, subject.getMeanAbsoluteDeviation());
    assertNotEquals(0.0, subject.getMeanAbsolutePercentageError());
    assertNotEquals(0.0, subject.getMeanSquaredError());
    assertNotEquals(0.0, subject.getTotalAbsoluteError());
    assertNotEquals(0.0, subject.getTotalSquaredError());
  }
}
