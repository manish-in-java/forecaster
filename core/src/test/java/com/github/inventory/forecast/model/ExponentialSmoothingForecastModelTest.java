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
import com.github.inventory.forecast.domain.Sample;
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
    // Generate a sample of random values.
    final int samples = getSampleCount();
    final double[] observations = new double[samples];

    for (int i = 0; i < samples; ++i)
    {
      observations[i] = getDouble();
    }

    // Generate a forecast for the sample.
    final Forecast subject = model.forecast(new Sample(observations), getProjectionCount());
    final double[] predictions = subject.getPredictions();

    assertNotNull(subject);
    assertNotNull(predictions);
    assertTrue(predictions.length > samples);

    for (int i = 0; i < observations.length; ++i)
    {
      assertNotEquals(0.0, predictions[i]);
    }

    // Ensure that the accuracy measures have been calculated.
    assertNotEquals(0.0, subject.getBias());
    assertNotEquals(0.0, subject.getMeanAbsoluteDeviation());
    assertNotEquals(0.0, subject.getMeanAbsolutePercentageError());
    assertNotEquals(0.0, subject.getMeanSquaredError());
    assertNotEquals(0.0, subject.getTotalAbsoluteError());
    assertNotEquals(0.0, subject.getTotalSquaredError());
  }
}
