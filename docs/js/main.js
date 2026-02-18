// Language tab switching
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.lang-tabs').forEach(function(tabGroup) {
    var panels = tabGroup.parentElement.querySelectorAll('.lang-panel');
    tabGroup.querySelectorAll('.lang-tab').forEach(function(tab) {
      tab.addEventListener('click', function() {
        var lang = this.dataset.lang;
        tabGroup.querySelectorAll('.lang-tab').forEach(function(t) { t.classList.remove('active'); });
        this.classList.add('active');
        panels.forEach(function(p) {
          p.classList.toggle('active', p.dataset.lang === lang);
        });
      });
    });
  });

  // Highlight current page in sidebar
  var path = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.sidebar a').forEach(function(link) {
    var href = link.getAttribute('href');
    if (href === path || (path === '' && href === 'index.html')) {
      link.classList.add('active');
    }
  });
});
