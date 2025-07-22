// Thought Generator Logic
export const thoughts = [
  "Learning structure before precision.",
  "Latent space isn’t just a trick — it’s where abstraction lives.",
  "Every diffusion path is a story, not just noise.",
  "Generative models are scientific hypotheses, not black boxes.",
  "Imagination through math: that’s generative AI.",
  "VAEs are microscopes for hidden representations.",
  "Can priors become intelligent agents?",
  "What if loss functions were emergent?",
  "The future of science is generative.",
  "Latent spaces are compressed belief systems."
];

const thoughtText = document.getElementById("thought-text");

if (thoughtText) {
  let index = 0;

  function rotateThought() {
    thoughtText.classList.add("fade-out");
    setTimeout(() => {
      index = (index + 1) % thoughts.length;
      thoughtText.textContent = `"${thoughts[index]}"`;
      thoughtText.classList.remove("fade-out");
    }, 600);
  }

  setInterval(rotateThought, 6000);
}
