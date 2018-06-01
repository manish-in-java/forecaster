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

import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;

/**
 * Generates a forecast for a sample using a weighted average algorithm
 * that uses exponentially decreasing weights as the observations in a
 * given sample get older. This gives more weight to more recent observations
 * and lesser to older ones, amplifying the recent trends and dampening or
 * smoothing the older ones.
 */
abstract class ExponentialSmoothingForecastModel extends ForecastModel
{
  static final double   DAMPENING_FACTOR_CONVERGENCE_THRESHOLD = 1e-6;
  static final double   INITIAL_DAMPENING_FACTOR               = 0.5;
  static final double   MAX_DAMPENING_FACTOR                   = 1.0;
  static final int      MAX_OPTIMIZATION_EVALUATIONS           = 100;
  static final int      MAX_OPTIMIZATION_ITERATIONS            = 100;
  static final double   MIN_DAMPENING_FACTOR                   = 0.0;
  static final GoalType OPTIMIZATION_GOAL                      = GoalType.MINIMIZE;
}
