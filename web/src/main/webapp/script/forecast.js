;(function($) {
  var dataSets = [
    "118, 93, 153, 125, 102, 141, 113, 99, 180, 162, 122, 181, 170, 143, 185, 195, 162, 205, 212, 162, 205, 184, 196, 249"
    , "30, 21, 29, 31, 40, 48, 53, 47, 37, 39, 31, 29, 17, 9, 20, 24, 27, 35, 41, 38, 27, 31, 27, 26, 21, 13, 21, 18, 33, 35, 40, 36, 22, 24, 21, 20, 17, 14, 17, 19, 26, 29, 40, 31, 20, 24, 18, 26, 17, 9, 17, 21, 28, 32, 46, 33, 23, 28, 22, 27, 18, 8, 17, 21, 31, 34, 44, 38, 31, 30, 26, 32"
    , "362,  385,  432,  341,  382,  409,  498,  387,  473,  513,  582,  474,  544,  582,  681,  557,  628,  707,  773,  592,  627,  725,  854,  661"
    , "112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432"
    , "1125, 1177, 1224, 1264, 1326, 1367, 1409, 1456, 1500, 1570, 1636, 1710, 1440, 1493, 1553, 1611, 1674, 1742, 1798, 1876, 1955, 2033, 2115, 2190, 1955, 2022, 2117, 2216, 2295, 2403, 2498, 2602, 2723, 2837, 2948, 3066"
  ];

  /*
   * Intercepts selection of radio buttons and loads associated sample data
   * set into the sample.
   */
  initializeDataSetSelection = function() {
    $(":radio").click(function() {
      $("#sampleData").val(dataSets[$(this).val()]);
    });
  }

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
                  animation       : { duration : 0 }
                  , scales        : {
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
      initializeDataSetSelection();
    });
  }
})(jQuery);
