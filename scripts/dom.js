export async function loadComponent(id, path) {
  const el = document.getElementById(id);
  if (!el) return;
  const res = await fetch(path);
  const html = await res.text();
  el.innerHTML = html;
}