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
import org.junit.Test;

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
    MODEL = new TripleExponentialSmoothingForecastModel(PERIODS, false);
  }

  /**
   * Tests that the triple exponential smoothing model cannot be constructed
   * without specifying a natural number of periods per season.
   */
  @Test(expected = IllegalArgumentException.class)
  public void testConstructWithNegativePeriod()
  {
    new TripleExponentialSmoothingForecastModel(-1, false);
  }

  /**
   * Tests that the triple exponential smoothing model cannot be used to
   * generate a forecast if the number of observations provided to it is not
   * a whole number multiple of the number of periods per season.
   */
  @Test(expected = IllegalArgumentException.class)
  public void testForecastWithoutFullSeason()
  {
    getForecastModel().forecast(getSample(2 * PERIODS + 1), getProjectionCount());
  }

  /**
   * Tests that the triple exponential smoothing model cannot be used to
   * generate a forecast if the number of observations provided to it is
   * not adequate to cover at least two seasons.
   */
  @Test(expected = IllegalArgumentException.class)
  public void testForecastWithoutTwoSeasons()
  {
    getForecastModel().forecast(getSample(PERIODS), getProjectionCount());
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
  int getSampleSize()
  {
    return PERIODS * getInt(3, 5);
  }
}
