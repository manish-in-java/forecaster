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

package com.github.inventory.web.controller;

import com.github.inventory.forecast.domain.Forecast;
import com.github.inventory.forecast.domain.Sample;
import com.github.inventory.forecast.model.ForecastModel;
import com.github.inventory.forecast.model.NaiveForecastModel;
import com.github.inventory.forecast.model.SimpleAverageForecastModel;
import com.github.inventory.forecast.model.WeightedAverageForecastModel;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.Arrays;

/**
 * Forecast generator controller.
 */
@Controller
@RequestMapping("/forecast")
public class ForecastController
{
  /**
   * Generates a forecast based on sample data and choice of a forecasting
   * model.
   */
  @PostMapping
  @ResponseBody
  public SampleAndForecast forecast(final ForecastModelEnum forecastingModel, final String sampleData, final Model model)
  {
    final Sample sample = getSample(sampleData);

    return new SampleAndForecast(sample, forecastingModel.getModel().forecast(sample));
  }

  /**
   * Displays the Forecast page.
   */
  @GetMapping
  public String show()
  {
    return "forecast";
  }

  /**
   * Gets a sample of data points presented as delimited values, such as,
   * {@literal 1, 2, 3, 4, 5} or {@literal 1;2;3;4;5}, etc.
   *
   * @param data The data points for the sample.
   * @return A {@link Sample}.
   */
  private Sample getSample(final String data)
  {
    final String[] dataPoints = data != null ? data.split(",|;| |\\|") : new String[0];

    final Sample sample = new Sample();

    Arrays.stream(dataPoints)
          .filter(dataPoint -> dataPoint != null && !"".equals(dataPoint.trim()))
          .mapToDouble(Double::parseDouble)
          .forEach(sample::add);

    return sample;
  }

  /**
   * Represents a model to use for generating a forecast.
   */
  private enum ForecastModelEnum
  {
    NAIVE(new NaiveForecastModel()),

    SIMPLE_AVERAGE(new SimpleAverageForecastModel()),

    WEIGHTED_AVERAGE(new WeightedAverageForecastModel(new double[] { 4, 3, 2, 1 }));

    private final ForecastModel model;

    /**
     * Sets the forecast model to use.
     *
     * @param model A {@link ForecastModel}.
     */
    ForecastModelEnum(final ForecastModel model)
    {
      this.model = model;
    }

    /**
     * Gets the forecast model corresponding to use for generating a forecast.
     *
     * @return A {@link ForecastModel}.
     */
    ForecastModel getModel()
    {
      return model;
    }
  }

  /**
   * A container for sample and forecast data.
   */
  private class SampleAndForecast
  {
    private final Forecast forecast;
    private final Sample   sample;

    /**
     * Sets the sample and its corresponding forecast data.
     *
     * @param sample   A {@link Sample}.
     * @param forecast A {@link Forecast}.
     */
    private SampleAndForecast(final Sample sample, final Forecast forecast)
    {
      this.forecast = forecast;
      this.sample = sample;
    }

    /**
     * Gets a forecast.
     *
     * @return A {@link Forecast}.
     */
    public Forecast getForecast()
    {
      return forecast;
    }

    /**
     * Gets the sample data.
     *
     * @return A {@link Sample}.
     */
    public Sample getSample()
    {
      return sample;
    }
  }
}
