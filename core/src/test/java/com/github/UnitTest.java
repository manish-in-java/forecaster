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

package com.github;

import java.security.SecureRandom;
import java.util.Random;

/**
 * Contract and utility methods for a unit test.
 */
public interface UnitTest
{
  Random RANDOM = new SecureRandom();

  /**
   * Gets a random double value between {@code 1,000} and {@code 1,000,000},
   * both inclusive.
   *
   * @return A random double value between {@code 1,000} and {@code 1,000,000}.
   */
  default double getDouble()
  {
    return getDouble(1000, 1000000);
  }

  /**
   * Gets a random double value between a minimum and a maximum value,
   * including the specified maximum and minimum values.
   *
   * @param minimum The minimum value.
   * @param maximum The maximum value.
   * @return A random value between the {@code minimum} and {@code maximum},
   * both inclusive.
   */
  default double getDouble(final double minimum, final double maximum)
  {
    return minimum + RANDOM.nextInt((int) (maximum - minimum + 1));
  }

  /**
   * Gets a random integer between {@code 1} and {@code 10}, both inclusive.
   *
   * @return A random integer between {@code 1} and {@code 10}.
   */
  default int getInt()
  {
    return getInt(1, 10);
  }

  /**
   * Gets a random integer between a minimum and a maximum value, including the
   * specified maximum and minimum values.
   *
   * @param minimum The minimum value.
   * @param maximum The maximum value.
   * @return A random integer between the {@code minimum} and {@code maximum},
   * both inclusive.
   */
  default int getInt(final int minimum, final int maximum)
  {
    return minimum + RANDOM.nextInt(maximum - minimum + 1);
  }
}
