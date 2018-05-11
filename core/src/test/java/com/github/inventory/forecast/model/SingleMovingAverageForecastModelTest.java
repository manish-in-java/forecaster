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
    // Generate a sample of random values.
    final int samples = getInt(11, 20);
    final Sample sample = new Sample(samples);

    for (int i = 0; i < samples; ++i)
    {
      sample.add(getDouble());
    }

    // Generate a forecast for the sample.
    final Forecast subject = getForecastModel().forecast(sample, getInt(5, 10));

    assertNotNull(subject);
    assertFalse(subject.isEmpty());
    assertTrue(subject.size() > samples);

    for (int i = 0; i < subject.size(); ++i)
    {
      if (i < GROUP - 1)
      {
        // The first few predictions must be undefined because there won't
        // be enough observations to generate the moving average.
        assertEquals(0d, subject.get(i), 0);
      }
      else
      {
        assertNotEquals(0d, subject.get(i));
      }
    }

    // Ensure that the accuracy measures have been calculated.
    assertNotEquals(0d, subject.getBias());
    assertNotEquals(0d, subject.getMeanAbsoluteDeviation());
    assertNotEquals(0d, subject.getMeanAbsolutePercentageError());
    assertNotEquals(0d, subject.getMeanSquaredError());
    assertNotEquals(0d, subject.getTotalAbsoluteError());
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
