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

import org.junit.BeforeClass;

/**
 * Unit tests for {@link TripleExponentialSmoothingForecastModel}.
 */
public class TripleExponentialSmoothingForecastModelTest extends ExponentialSmoothingForecastModelTest
{
  private static       TripleExponentialSmoothingForecastModel MODEL;
  private static final int                                     PERIODS = 4;

  /**
   * Sets up objects required to run the tests.
   */
  @BeforeClass
  public static void setup()
  {
    MODEL = new TripleExponentialSmoothingForecastModel(PERIODS);
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
   * {@inheritDoc}
   */
  @Override
  int getSampleCount()
  {
    return PERIODS * getInt(2, 5);
  }
}
