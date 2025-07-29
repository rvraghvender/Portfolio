fetch("assets/publications/publications_records.json")
    .then((response) => response.json())
    .then((data) => {
        const container = document.getElementById("publications-index");
        const pubList = document.createElement("div");
        pubList.className = "publications-container";

        data.forEach((pub) => {
            const authors = pub.authors.map(name =>
                name.toLowerCase().includes("raghvender") ? `<strong>${name}</strong>` : name
            ).join(", ");

            const card = `
        <div class="publication-card">
          <div class="pub-meta">

            <h2 class="pub-title">
            <a href="${pub.doi_link}" target="_blank" rel="noopener noreferrer">${pub.title}</a>
            </h2>

            <p class="pub-authors">${authors}</p>
            <p class="pub-venue">${pub.venue}</p>
          </div>
          <details class="pub-abstract">
            <summary>Abstract</summary>
            <p>${pub.abstract}</p>
          </details>
        </div>
      `;
            pubList.innerHTML += card;
        });

        container.appendChild(pubList);
    })
    .catch((err) => console.error("Failed to load publications:", err));
