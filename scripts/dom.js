export async function loadComponent(id, path) {
  const el = document.getElementById(id);
  if (!el) return;

  const res = await fetch(path);
  const html = await res.text();
  el.innerHTML = html;

  // After header is loaded, wire up mobile menu
  if (id === 'header') {
    const hamburger = document.getElementById('hamburger');
    const mobileMenu = document.getElementById('mobileMenu');

    if (hamburger && mobileMenu) {
      hamburger.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
      });
    }
  }

  // Same idea if you want footer/sidebar interactivity later
}
