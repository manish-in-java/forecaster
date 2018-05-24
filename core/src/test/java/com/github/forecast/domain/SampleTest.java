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
 * Unit tests for {@link Sample}.
 */
public class SampleTest implements UnitTest
{
  /**
   * Tests that a sample can be constructed by specifying the observations
   * to include.
   */
  @Test
  public void testConstruct()
  {
    // Generate a random set of observations.
    final double[] observations = new double[getInt()];
    for (int i = 0; i < observations.length; ++i)
    {
      observations[i] = getDouble();
    }

    final double[] subject = new Sample(observations).getObservations();

    // Verify that the sample returns a copy of the observations.
    assertNotSame(observations, subject);
    assertEquals(observations.length, subject.length);
    for (int i = 0; i < observations.length; ++i)
    {
      assertEquals(observations[i], subject[i], 0.0);
    }
  }

  /**
   * Tests that a sample cannot be created with no observations.
   */
  @Test(expected = IllegalArgumentException.class)
  public void testConstructWithNoObservations()
  {
    new Sample(new double[0]);
  }

  /**
   * Tests that a sample cannot be created without specifying observations.
   */
  @Test(expected = NullPointerException.class)
  public void testConstructWithoutObservations()
  {
    new Sample(null);
  }
}
