export async function loadComponent(id, path) {
  const el = document.getElementById(id);
  if (!el) return;

  const res = await fetch(path);
  const html = await res.text();
  el.innerHTML = html;

  if (id === 'header') {
    const hamburger = document.getElementById('hamburger');
    const mobileMenu = document.getElementById('mobileMenu');

    if (hamburger && mobileMenu) {
      hamburger.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
      });
    }

    // ✅ Add dark mode toggle here
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

    const savedPreference = localStorage.getItem('darkMode') === 'true';
    if (savedPreference) {
      body.classList.add('dark-mode');
    }

    updateThemeIcon(savedPreference);
  }
}
