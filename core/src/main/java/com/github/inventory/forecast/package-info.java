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

/**
 * <p>
 * Forecasting is important for businesses to plan which products to carry in
 * the inventory, how much quantity to carry and where should the quantity
 * be carried (stored). Good forecasts enable businesses to reduce the cost
 * of carrying inventory, while avoiding out-of-stock scenarios for
 * in-demand products.
 * </p>
 * <p>
 * Forecasting is an estimation process that is affected by a number of
 * factors, including the nature of the product, seasonality, customer
 * preferences, socioeconomic environment, competitor dynamics, etc.
 * </p>
 * <p>
 * Forecasting is usually performed using actual data collected in the past.
 * For example, a sales forecast can be performed using past sales data.
 * Similarly, a price forecast for a stock traded on a stock exchange can be
 * performed using the historical prices for the stock. Each data point is
 * called an {@literal Observation} and the collection of observations, based
 * on which the forecast is generated a {@literal sample}.
 * </p>
 * <p>
 * There are many forecasting techniques. Each technique, known as a
 * {@literal Forecast Model} (or {@literal Model} in short) attempts to
 * analyze the sample in a certain way so as to be as accurate as possible.
 * Each model amplifies or suppresses one or more pattern underlying the sample
 * set. As a result, different types of samples benefit from different
 * forecast models.
 * </p>
 */
package com.github.inventory.forecast;
