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
 * Unit tests for {@link SimpleAverageForecastModel}.
 */
public class SimpleAverageForecastModelTest extends AverageForecastModelTest
{
  private static SimpleAverageForecastModel MODEL;

  /**
   * Sets up objects required to run the tests.
   */
  @BeforeClass
  public static void setup()
  {
    MODEL = new SimpleAverageForecastModel();
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
