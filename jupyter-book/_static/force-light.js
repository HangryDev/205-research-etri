// Force light mode — prevent dark theme from activating
document.addEventListener('DOMContentLoaded', function() {
  document.documentElement.setAttribute('data-theme', 'light');
  localStorage.setItem('data-theme', 'light');
});
