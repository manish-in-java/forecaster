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
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Unit tests for {@link SingleMovingAverageForecastModel}.
 */
public class SingleMovingAverageForecastModelTest extends ForecastModelTest
{
  private static int                              GROUP;
  private static SingleMovingAverageForecastModel MODEL;

  /**
   * Sets up objects required to run the tests.
   */
  @BeforeClass
  public static void setup()
  {
    GROUP = 4;
    MODEL = new SingleMovingAverageForecastModel(GROUP);
  }

  /**
   * Tests that the single moving average forecast can be correctly generated
   * for a sample.
   */
  @Test
  public void testForecast()
  {
    // Generate a random sample of observations.
    final Sample sample = getSample();

    // Generate a forecast for the sample.
    final Forecast subject = getForecastModel().forecast(sample, getProjectionCount());
    final double[] predictions = subject.getPredictions();

    assertNotNull(subject);
    assertNotNull(predictions);
    assertTrue(predictions.length > sample.size());

    for (int i = 0; i < predictions.length; ++i)
    {
      if (i < GROUP - 1)
      {
        // The first few predictions must be undefined because there won't
        // be enough observations to generate the moving average.
        assertEquals(0.0, predictions[i], 0);
      }
      else
      {
        assertNotEquals(0.0, predictions[i]);
      }
    }

    // Ensure that the accuracy measures have been calculated.
    assertNotEquals(0.0, subject.getBias());
    assertNotEquals(0.0, subject.getMeanAbsoluteDeviation());
    assertNotEquals(0.0, subject.getMeanAbsolutePercentageError());
    assertNotEquals(0.0, subject.getMeanSquaredError());
    assertNotEquals(0.0, subject.getTotalAbsoluteError());
    assertNotEquals(0.0, subject.getTotalSquaredError());
  }

  /**
   * {@inheritDoc}
   */
  @Override
  ForecastModel getForecastModel()
  {
    return MODEL;
  }
}
