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
import com.github.inventory.forecast.domain.Observation;
import com.github.inventory.forecast.domain.Sample;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Unit tests for {@link SimpleAverageForecastingModel}.
 */
public class SimpleAverageForecastingModelTest extends ForecastingModelTest
{
  private static SimpleAverageForecastingModel MODEL;

  /**
   * Sets up objects required to run the tests.
   */
  @BeforeClass
  public static void setup()
  {
    MODEL = new SimpleAverageForecastingModel();
  }

  /**
   * Tests that the simple average forecast can be correctly generated for
   * a sample.
   */
  @Test
  public void testForecast()
  {
    // Generate a sample of random values.
    final int samples = getInt(11, 20);
    final Sample sample = new Sample(samples);

    for (int i = 0; i < samples; ++i)
    {
      sample.add(new Observation(getDouble()));
    }

    // Generate a forecast for the sample.
    final Forecast subject = getForecastingModel().forecast(sample, getInt(5, 10));

    assertNotNull(subject);
    assertFalse(subject.isEmpty());
    assertTrue(subject.size() > samples);

    subject.forEach(prediction -> {
      assertNotNull(prediction);
      assertNotEquals(0, prediction.getMeanSquaredError());
      assertNotEquals(0, prediction.getValue());
      assertTrue(prediction.isDefined());
      assertNotNull(prediction.toString());
    });
  }

  /**
   * {@inheritDoc}
   */
  @Override
  ForecastingModel getForecastingModel()
  {
    return MODEL;
  }
}
