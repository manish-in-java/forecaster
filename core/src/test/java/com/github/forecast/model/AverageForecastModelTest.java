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
 * Unit tests for forecast models that use some form of averaging to make
 * predictions for a sample.
 */
abstract class AverageForecastModelTest extends ForecastModelTest
{
  /**
   * Tests that the simple average forecast can be correctly generated for
   * a sample.
   */
  @Test
  public void testForecast()
  {
    // Generate a sample of random values.
    final int samples = getSampleCount();
    final double[] observations = new double[samples];

    for (int i = 0; i < samples; ++i)
    {
      observations[i] = getDouble();
    }

    // Generate a forecast for the sample.
    final Forecast subject = getForecastModel().forecast(new Sample(observations), getProjectionCount());
    final double[] predictions = subject.getPredictions();

    assertNotNull(subject);
    assertNotNull(predictions);
    assertTrue(predictions.length > samples);

    for (final double prediction : predictions)
    {
      assertNotEquals(0.0, prediction);
    }

    // Ensure that the accuracy measures have been calculated.
    assertNotEquals(0.0, subject.getMeanAbsoluteDeviation());
    assertNotEquals(0.0, subject.getMeanAbsolutePercentageError());
    assertNotEquals(0.0, subject.getMeanSquaredError());
    assertNotEquals(0.0, subject.getTotalAbsoluteError());
    assertNotEquals(0.0, subject.getTotalSquaredError());

    // The following test should not be performed for averaging models
    // as the bias could be zero if the observations are equally distributed
    // on either side of the average.
    // assertNotEquals(0.0, subject.getBias());
  }
}
