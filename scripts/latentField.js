const canvas = document.getElementById('latentField');
const ctx = canvas?.getContext('2d');

const NUM_PARTICLES = Math.floor(window.innerWidth / 10);


if (canvas && ctx) {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const nodes = Array.from({ length: NUM_PARTICLES }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    vx: (Math.random() - 0.5) * 0.4,
    vy: (Math.random() - 0.5) * 0.4
  }));

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#dde1e9ff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#494b4eff';
    ctx.strokeStyle = '#404142ff';
    ctx.lineWidth = 0.8;

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      ctx.beginPath();
      ctx.arc(node.x, node.y, 2, 0, 2 * Math.PI);
      ctx.fill();

      for (let j = i + 1; j < nodes.length; j++) {
        const other = nodes[j];
        const dx = node.x - other.x;
        const dy = node.y - other.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 120) {
          ctx.globalAlpha = 1 - dist / 120;
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(other.x, other.y);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  function update() {
    for (const node of nodes) {
      node.x += node.vx;
      node.y += node.vy;

      if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
      if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
    }
  }

  function animate() {
    update();
    draw();
    requestAnimationFrame(animate);
  }

  animate();

  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  });
}

// const isDarkMode = document.body.classList.contains('dark-mode');
// ctx.fillStyle = isDarkMode ? '#0f172a' : '#f9fafb';
// ctx.strokeStyle = isDarkMode ? '#64748b' : '#a0aec0';
