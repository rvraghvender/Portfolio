import { loadComponent } from './dom.js';
import { thoughts } from './thoughts.js';


// Load HTML components
loadComponent('header', '/components/header.html');
loadComponent('footer', '/components/footer.html');

// Load sidebar only if it exists (not on homepage)
const sidebarContainer = document.getElementById('sidebar');
if (sidebarContainer) {
  loadComponent('sidebar', '/components/sidebar.html');
}

// Theme Toggle Setup
const toggleButton = document.getElementById('theme-toggle');
const body = document.body;

function updateThemeIcon(isDark) {
  if (toggleButton) {
    toggleButton.textContent = isDark ? '☀️' : '🌙';
  }
}

if (toggleButton) {
  toggleButton.addEventListener('click', () => {
    body.classList.toggle('dark-mode');
    const isDark = body.classList.contains('dark-mode');
    updateThemeIcon(isDark);
    localStorage.setItem('darkMode', isDark);
  });
}

// Load saved preference
const savedPreference = localStorage.getItem('darkMode') === 'true';
if (savedPreference) {
  body.classList.add('dark-mode');
}
updateThemeIcon(savedPreference);

