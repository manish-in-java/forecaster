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
 * Unit tests for {@link WeightedAverageForecastModel}.
 */
public class WeightedAverageForecastModelTest extends ForecastModelTest
{
  private static WeightedAverageForecastModel MODEL;
  private static double[]                     WEIGHTS;

  /**
   * Sets up objects required to run the tests.
   */
  @BeforeClass
  public static void setup()
  {
    WEIGHTS = new double[] { 0.4, 0.3, 0.2, 0.1 };
    MODEL = new WeightedAverageForecastModel(WEIGHTS);
  }

  /**
   * Tests that a weighted average forecast model can be created with
   * weights whose sum exceeds {@literal one}.
   */
  @Test
  public void testConstructWithDenormalizedWeights()
  {
    final WeightedAverageForecastModel subject = new WeightedAverageForecastModel(new double[] { 1.0, 2.0, 4.0, 8.0 });

    final double[] weights = subject.getWeights();

    assertNotNull(weights);
    assertEquals(4, weights.length);
    for (final double weight : weights)
    {
      assertTrue(weight < 1.0);
    }
  }

  /**
   * Tests that a weighted average forecast model can be created with a
   * single weight.
   */
  @Test
  public void testConstructWithSingleWeight()
  {
    final WeightedAverageForecastModel subject = new WeightedAverageForecastModel(new double[] { getDouble() });

    final double[] weights = subject.getWeights();

    assertNotNull(weights);
    assertEquals(1, weights.length);
    assertEquals(1.0, weights[0], 0.0);
  }

  /**
   * Tests that a weighted average forecast model can be created without
   * specifying the weights and that doing so sets a fixed weight.
   */
  @Test
  public void testConstructWithoutWeights()
  {
    final WeightedAverageForecastModel subject = new WeightedAverageForecastModel(null);

    final double[] weights = subject.getWeights();

    assertNotNull(weights);
    assertEquals(1, weights.length);
    assertEquals(1.0, weights[0], 0.0);
  }

  /**
   * Tests that the weighted average forecast can be correctly generated for
   * a sample.
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
      if (i < WEIGHTS.length - 1)
      {
        // The first few predictions must be undefined because there won't
        // be enough observations to generate the weighted average.
        assertEquals(0.0, predictions[i], 0.0);
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
   * Tests that when observations are fewer than the weights, the model
   * skips observations until there are sufficient enough to generate a
   * prediction.
   */
  @Test
  public void testForecastWithExtraWeights()
  {
    // Use a large number of weights.
    final double[] weights = new double[] { 0.1, 0.2, 0.3, 0.4 };

    final WeightedAverageForecastModel model = new WeightedAverageForecastModel(weights);

    // Use fewer observations than the available weights.
    final double[] observations = new double[] { getDouble(), getDouble() };

    // Generate a forecast using the known weights and observations.
    final Forecast forecast = model.forecast(new Sample(observations));

    final double[] predictions = forecast.getPredictions();

    assertNotNull(predictions);
    assertEquals(observations.length + 1, predictions.length);
    for (int i = 0; i < observations.length; ++i)
    {
      assertEquals(0.0, predictions[i], 0.0);
    }
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
