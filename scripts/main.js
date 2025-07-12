import { loadComponent } from './dom.js';
loadComponent('header', '/components/header.html');
loadComponent('footer', '/components/footer.html');
loadComponent('sidebar', '/components/sidebar.html');


const toggleButton = document.getElementById('theme-toggle');
const body = document.body;

function updateThemeIcon(isDark) {
  toggleButton.textContent = isDark ? '☀️' : '🌙';
}

toggleButton.addEventListener('click', () => {
  body.classList.toggle('dark-mode');
  const isDark = body.classList.contains('dark-mode');
  updateThemeIcon(isDark);
  localStorage.setItem('darkMode', isDark);
});

// Load user preference on page load
const savedPreference = localStorage.getItem('darkMode') === 'true';
if (savedPreference) {
  body.classList.add('dark-mode');
}
updateThemeIcon(savedPreference);
