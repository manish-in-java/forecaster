;(function($) {
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
      initializeMessageDismissalButtons();
    });
  }
})(jQuery);
