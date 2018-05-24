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

package com.github.forecast.domain;

import com.github.UnitTest;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;

/**
 * Unit tests for {@link Forecast}.
 */
public class ForecastTest implements UnitTest
{
  /**
   * Tests that a forecast can be constructed by specifying the observations
   * and predictions to include.
   */
  @Test
  public void testConstruct()
  {
    // Generate a random set of observations and predictions.
    final double[] observations = new double[getInt()];
    final double[] predictions = new double[observations.length];
    for (int i = 0; i < observations.length; ++i)
    {
      observations[i] = getDouble();
      predictions[i] = getDouble();
    }

    final double[] subject = new Forecast(observations, predictions).getPredictions();

    // Verify that the forecast returns a copy of the predictions.
    assertNotSame(predictions, subject);
    assertEquals(predictions.length, subject.length);
    for (int i = 0; i < predictions.length; ++i)
    {
      assertEquals(predictions[i], subject[i], 0.0);
    }
  }

  /**
   * Tests that a forecast cannot be created with no observations.
   */
  @Test(expected = IllegalArgumentException.class)
  public void testConstructWithNoObservations()
  {
    new Forecast(new double[0], null);
  }

  /**
   * Tests that a forecast cannot be created with no predictions.
   */
  @Test(expected = IllegalArgumentException.class)
  public void testConstructWithNoPredictions()
  {
    new Forecast(new double[1], new double[0]);
  }

  /**
   * Tests that a forecast cannot be created without specifying observations.
   */
  @Test(expected = NullPointerException.class)
  public void testConstructWithoutObservations()
  {
    new Forecast(null, null);
  }

  /**
   * Tests that a forecast cannot be created without specifying predictions.
   */
  @Test(expected = NullPointerException.class)
  public void testConstructWithoutPredictions()
  {
    new Forecast(new double[1], null);
  }
}
