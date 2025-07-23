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
