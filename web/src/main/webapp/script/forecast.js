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
              var observations = response.sample.observations;

              // Extract predictions from the response.
              var options = new Array();
              var predictions = response.forecast.predictions;
              for(var i = 0; i < predictions.length; ++i) {
                options.push(i + 1);
              }

              // Display a line chart with the sample data.
              new Chart($("#chart")
              , {
                data              : {
                  datasets        : [{
                    borderColor   : "#FFC800"
                    , data        : observations
                    , fill        : false
                    , lineTension : 0.1
                    , label       : "Data points"
                  }, {
                    borderColor   : "#009F8B"
                    , data        : predictions
                    , fill        : false
                    , lineTension : 0.1
                    , label       : "Forecast"
                  }]
                  , labels        : options
                }
                , options         : {
                  scales          : {
                    yAxes         : [{
                      ticks       : { beginAtZero : true }
                    }]
                  }
                }
                , type            : "line"
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
