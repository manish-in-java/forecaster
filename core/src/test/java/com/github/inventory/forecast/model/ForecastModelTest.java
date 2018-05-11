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

import com.github.UnitTest;
import com.github.inventory.forecast.domain.Sample;
import org.junit.Test;

import static org.junit.Assert.assertNull;

/**
 * Unit tests for {@link ForecastModel}.
 */
abstract class ForecastModelTest implements UnitTest
{
  /**
   * Tests that a forecast cannot be obtained with empty sample.
   */
  @Test
  public void testForecastWithEmptySampleSet()
  {
    assertNull(getForecastModel().forecast(getEmptySample()));
  }

  /**
   * Tests that a forecast cannot be obtained without specifying the sample.
   */
  @Test
  public void testForecastWithNullSampleSet()
  {
    assertNull(getForecastModel().forecast(null));
  }

  /**
   * Gets a {@link ForecastModel} for the tests.
   *
   * @return A {@link ForecastModel}.
   */
  abstract ForecastModel getForecastModel();

  /**
   * Gets empty sample for the tests.
   *
   * @return Empty sample.
   */
  private Sample getEmptySample()
  {
    return new Sample();
  }
}
