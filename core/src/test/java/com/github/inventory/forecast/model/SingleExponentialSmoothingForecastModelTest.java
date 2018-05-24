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
import com.github.inventory.forecast.model.SingleExponentialSmoothingForecastModel.AlphaOptimizer;
import com.github.inventory.forecast.model.SingleExponentialSmoothingForecastModel.FirstPredictionGenerator;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Unit tests for {@link SingleExponentialSmoothingForecastModel}.
 */
public class SingleExponentialSmoothingForecastModelTest extends ForecastModelTest
{
  private static SingleExponentialSmoothingForecastModel MODEL;

  /**
   * Sets up objects required to run the tests.
   */
  @BeforeClass
  public static void setup()
  {
    MODEL = new SingleExponentialSmoothingForecastModel();
  }

  /**
   * Tests that the single exponential smoothing forecast can be correctly
   * generated for a sample.
   */
  @Test
  public void testForecast()
  {
    testForecast(getForecastModel());
  }

  /**
   * Tests that a forecast model can be constructed with an alternate
   * strategy for generating the first prediction.
   */
  @Test
  public void testForecastWithAlternateFirstPredictionGenerator()
  {
    testForecast(new SingleExponentialSmoothingForecastModel(FirstPredictionGenerator.FIRST_OBSERVATION, AlphaOptimizer.GRADIENT_DESCENT));
  }

  /**
   * Tests that a forecast model cannot be constructed without specifying
   * the strategy for optimizing the value of the dampening factor.
   */
  @Test(expected = NullPointerException.class)
  public void testForecastWithoutAlphaOptimizer()
  {
    testForecast(new SingleExponentialSmoothingForecastModel(FirstPredictionGenerator.FIRST_OBSERVATION, null));
  }

  /**
   * Tests that a forecast model cannot be constructed without specifying
   * the strategy for generating the first prediction.
   */
  @Test(expected = NullPointerException.class)
  public void testForecastWithoutFirstPredictionGenerator()
  {
    testForecast(new SingleExponentialSmoothingForecastModel(null, AlphaOptimizer.GRADIENT_DESCENT));
  }

  /**
   * {@inheritDoc}
   */
  @Override
  ForecastModel getForecastModel()
  {
    return MODEL;
  }

  /**
   * Tests that the single exponential smoothing forecast can be correctly
   * generated for a sample.
   *
   * @param model The model to use for generating the forecast.
   */
  private void testForecast(final ForecastModel model)
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
