;(function($) {
  /*
   * Intercepts form submit to load forecast using an AJAX call and rendering
   * it as a chart, along with the sample data, for comparison.
   */
  initializeForecastRenderer = function() {
    var content = new Vue({
      el        : "#content"
      , data    : {
        error   : false
      }
      , methods : {
        forecast : function() {
          var form = $("#form");
          var forecastingModel = $("#forecastingModel");
          var sampleData = $("#sampleData");
          var trigger = $("#trigger");

          // Hide error message, if visible.
          content.error = false;

          // Start animation to indicate that the data is being loaded.
          trigger.addClass("loading");

          // Issue an AJAX request to load the forecast.
          $.ajax({
            cache  : false
            , data : {
              forecastingModel : forecastingModel.val()
              , sampleData     : sampleData.val()
            }
            , type : form.attr("method")
            , url  : form.attr("action")
            , error : function() {
              // Stop animation.
              trigger.removeClass("loading");

              // Display an error message.
              content.error = true;
            }
            , success : function(response) {
              // Stop animation.
              trigger.removeClass("loading");

              // Extract observations from the response.
              var observations = response.sample;

              // Extract predictions from the response.
              var options = new Array();
              var predictions = new Array();
              for(var i = 0; i < response.forecast.length; ++i) {
                options.push(i + 1);

                predictions.push(response.forecast[i].defined ? response.forecast[i].value : NaN);
              }

              // Display a line chart with the sample data.
              new Chart($("#chart")
              , {
                "data"        : {
                  "datasets"  : [{
                    "borderColor"   : "rgb(255, 99, 132)"
                    , "data"        : observations
                    , "fill"        : false
                    , "lineTension" : 0.1
                    , "label"       : "Sample"
                  }, {
                    "borderColor"   : "rgb(75, 192, 192)"
                    , "data"        : predictions
                    , "fill"        : false
                    , "lineTension" : 0.1
                    , "label"       : "Forecast"
                  }]
                  , "labels"  : options
                }
                , "type"      : "line"
              });
            }
          });
        }
      }
    });
  }

  if ($) {
    $(window).ready(function() {
      initializeForecastRenderer();
    });
  }
})(jQuery);
