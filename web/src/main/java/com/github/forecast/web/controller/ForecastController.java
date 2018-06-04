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

package com.github.forecast.web.controller;

import com.github.forecast.domain.Forecast;
import com.github.forecast.domain.Sample;
import com.github.forecast.model.*;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.Arrays;

/**
 * Forecast generator controller.
 */
@Controller
@RequestMapping("/")
public class ForecastController
{
  /**
   * Generates a forecast based on sample data and choice of a forecasting
   * model.
   */
  @PostMapping
  @ResponseBody
  public SampleAndForecast forecast(final ForecastModelEnum forecastingModel, final String sampleData)
  {
    final Sample sample = getSample(sampleData);

    return new SampleAndForecast(sample, forecastingModel.getModel().forecast(sample, forecastingModel.getProjections()));
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

    final double[] observations = Arrays.stream(dataPoints)
                                        .filter(dataPoint -> dataPoint != null && !"".equals(dataPoint.trim()))
                                        .mapToDouble(Double::parseDouble)
                                        .toArray();

    return new Sample(observations);
  }

  /**
   * Represents a model to use for generating a forecast.
   */
  private enum ForecastModelEnum
  {
    DES(new DoubleExponentialSmoothingForecastModel(), 4),

    EWMA(new ExponentialWeightedMovingAverageForecastModel()),

    NAIVE(new NaiveForecastModel()),

    SIAV(new SimpleAverageForecastModel()),

    SES(new SingleExponentialSmoothingForecastModel()),

    SMA(new SingleMovingAverageForecastModel(3)),

    STAV(new StraightAverageForecastModel()),

    TES4A(new TripleExponentialSmoothingForecastModel(4, false), 4),

    TES4M(new TripleExponentialSmoothingForecastModel(4, true), 4),

    TES7A(new TripleExponentialSmoothingForecastModel(7, false), 7),

    TES7M(new TripleExponentialSmoothingForecastModel(7, true), 7),

    TES12A(new TripleExponentialSmoothingForecastModel(12, false), 12),

    TES12M(new TripleExponentialSmoothingForecastModel(12, true), 12),

    WA(new WeightedAverageForecastModel(new double[] { 4, 3, 2, 1 }));

    private final ForecastModel model;
    private final int           projections;

    /**
     * Sets the forecast model to use.
     *
     * @param model A {@link ForecastModel}.
     */
    ForecastModelEnum(final ForecastModel model)
    {
      this(model, 1);
    }

    /**
     * Sets the forecast model to use and the number of projections to add
     * to the forecast.
     *
     * @param model       A {@link ForecastModel}.
     * @param projections The number of projections to add to the forecast.
     */
    ForecastModelEnum(final ForecastModel model, final int projections)
    {
      this.model = model;
      this.projections = projections;
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

    /**
     * Gets the number of projections to make beyond the sample size.
     *
     * @return The number of projections to make beyond the sample size.
     */
    int getProjections()
    {
      return projections;
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
