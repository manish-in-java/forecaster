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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Unit tests for {@link Prediction}
 */
public class PredictionTest implements UnitTest
{
  /**
   * Tests that a prediction with a defined predicted value is treated
   * as such.
   */
  @Test
  public void testIsDefinedWithDefined()
  {
    assertTrue(new Prediction(getDouble(), getDouble(), getDouble(), getDouble(), getDouble(), getDouble()).isDefined());
  }

  /**
   * Tests that a prediction with an undefined predicted value is treated
   * as such.
   */
  @Test
  public void testIsDefinedWithUndefined()
  {
    assertFalse(Prediction.undefined().isDefined());
  }
}
