;(function($) {
  /*
   * Attaches mouse event handlers for expanding and collapsing accordions.
   */
  initializeAccordions = function() {
    $(".ui.accordion").accordion();
  }

  /*
   * Attaches click handlers to buttons that can be used to dismiss
   * informational messages.
   */
  initializeMessageDismissalButtons = function() {
    $(".message .close").on("click", function() {
      $(this).closest(".message").transition("fade");
    });
  }

  if ($) {
    $(window).ready(function() {
      initializeAccordions();
      initializeMessageDismissalButtons();
    });
  }
})(jQuery);
