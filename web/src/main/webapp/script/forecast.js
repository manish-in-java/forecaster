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
          var chart = $("#chart");
          var form = $("#form");
          var forecastingModel = $("#forecastingModel");
          var sampleData = $("#sampleData");
          var trigger = $("#trigger");

          // Hide error message, if visible.
          content.error = true;

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
            , success : function(data) {
              // Stop animation.
              trigger.removeClass("loading");

              console.log(data);
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
