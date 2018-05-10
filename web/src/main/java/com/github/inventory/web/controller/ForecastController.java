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
import com.github.inventory.forecast.domain.Observation;
import com.github.inventory.forecast.domain.Sample;
import com.github.inventory.forecast.model.ForecastingModel;
import com.github.inventory.forecast.model.NaiveForecastingModel;
import com.github.inventory.forecast.model.SimpleAverageForecastingModel;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.Arrays;
import java.util.stream.Collectors;

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
  public Forecast forecast(final ForecastingModelEnum forecastingModel, final String data, final Model model)
  {
    return forecastingModel.getModel().forecast(getSample(data));
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
    final String[] dataPoints = data != null ? data.split(",; |") : new String[0];

    return Arrays.stream(dataPoints)
                 .filter(dataPoint -> dataPoint != null && !"".equals(dataPoint.trim()))
                 .mapToDouble(Double::parseDouble)
                 .mapToObj(Observation::new)
                 .collect(Collectors.toCollection(Sample::new));
  }

  /**
   * Represents a forecasting model to use for generating a forecast.
   */
  private enum ForecastingModelEnum
  {
    NAIVE(new NaiveForecastingModel()),

    SIMPLE_AVERAGE(new SimpleAverageForecastingModel());

    private final ForecastingModel model;

    /**
     * Sets the forecasting model to use.
     *
     * @param model A {@link ForecastingModel}.
     */
    ForecastingModelEnum(final ForecastingModel model)
    {
      this.model = model;
    }

    /**
     * Gets the forecasting model corresponding to use for generating a
     * forecast.
     *
     * @return A {@link ForecastingModel}.
     */
    ForecastingModel getModel()
    {
      return model;
    }
  }
}
