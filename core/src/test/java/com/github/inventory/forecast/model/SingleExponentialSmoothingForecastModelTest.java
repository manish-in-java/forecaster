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

import com.github.inventory.forecast.model.SingleExponentialSmoothingForecastModel.GradientDescentOptimizer;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Unit tests for {@link SingleExponentialSmoothingForecastModel}.
 */
public class SingleExponentialSmoothingForecastModelTest extends ExponentialSmoothingForecastModelTest
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
   * Tests that a forecast model can be constructed with an alternate
   * strategy for generating the first prediction.
   */
  @Test
  public void testForecastWithAlternateFirstPredictionGenerator()
  {
    testForecast(new SingleExponentialSmoothingForecastModel(SingleExponentialSmoothingForecastModel.SimpleFirstPredictionProvider.INSTANCE, GradientDescentOptimizer.INSTANCE));
  }

  /**
   * Tests that a forecast model cannot be constructed without specifying
   * the strategy for optimizing the value of the dampening factor.
   */
  @Test(expected = NullPointerException.class)
  public void testForecastWithoutAlphaOptimizer()
  {
    testForecast(new SingleExponentialSmoothingForecastModel(SingleExponentialSmoothingForecastModel.SimpleFirstPredictionProvider.INSTANCE, null));
  }

  /**
   * Tests that a forecast model cannot be constructed without specifying
   * the strategy for generating the first prediction.
   */
  @Test(expected = NullPointerException.class)
  public void testForecastWithoutFirstPredictionGenerator()
  {
    testForecast(new SingleExponentialSmoothingForecastModel(null, GradientDescentOptimizer.INSTANCE));
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
