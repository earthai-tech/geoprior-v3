(function () {
  "use strict";

  function onReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, {
        once: true,
      });
      return;
    }
    fn();
  }

  function qsAll(selector, root = document) {
    return Array.from(root.querySelectorAll(selector));
  }

  function normalizeText(text) {
    return (text || "")
      .trim()
      .replace(/\s+/g, " ")
      .toLowerCase();
  }

  function showRecentBadges() {
    const badges = qsAll(".new-badge, .new-badge-card");
    if (!badges.length) {
      return;
    }

    const now = Date.now();
    const visibleWindowMs = 7 * 24 * 60 * 60 * 1000;

    badges.forEach((badge) => {
      const releaseDateStr = badge.getAttribute("data-release-date");
      if (!releaseDateStr) {
        return;
      }

      const releaseDate = new Date(releaseDateStr);
      if (Number.isNaN(releaseDate.getTime())) {
        return;
      }

      const age = now - releaseDate.getTime();
      if (age >= 0 && age < visibleWindowMs) {
        badge.style.display = "inline-block";
      }
    });
  }

  function resolvePreviewImage(card) {
    const explicitPreview = card.dataset.preview;
    if (explicitPreview) {
      return explicitPreview;
    }

    const previewMap = {
      "card--uncertainty": "uncertainty.png",
      "card--errors": "errors.png",
      "card--evaluation": "evaluation.png",
      "card--importance": "importance.png",
      "card--relationship": "relationship.png",
      "card--physics": "physics.png",
      "card--workflow": "workflow.png",
      "card--cli": "cli.png",
      "card--configuration": "configuration.png",
    };

    for (const [className, filename] of Object.entries(previewMap)) {
      if (card.classList.contains(className)) {
        return filename;
      }
    }

    return null;
  }

  function buildPreviewPath(preview) {
    if (!preview) {
      return null;
    }
    if (/^(?:https?:)?\/\//.test(preview) || preview.startsWith("/")) {
      return preview;
    }
    return `_static/previews/${preview}`;
  }

  function setupCardPreviews() {
    const cards = qsAll(".seealso-card, .sd-card").filter((card) => {
      return Boolean(card.dataset.preview) ||
        Array.from(card.classList).some((cls) => cls.startsWith("card--"));
    });

    if (!cards.length) {
      return;
    }

    const previewNames = new Set();
    cards.forEach((card) => {
      const preview = resolvePreviewImage(card);
      if (preview) {
        previewNames.add(preview);
      }
    });

    previewNames.forEach((name) => {
      const img = new Image();
      img.src = buildPreviewPath(name);
    });

    cards.forEach((card) => {
      if (window.getComputedStyle(card).position === "static") {
        card.style.position = "relative";
      }

      const preview = resolvePreviewImage(card);
      const imagePath = buildPreviewPath(preview);
      if (!imagePath) {
        return;
      }

      const cardLabel =
        card.dataset.previewLabel ||
        card.getAttribute("aria-label") ||
        card.textContent.trim().slice(0, 80) ||
        "GeoPrior preview";

      let removalTimer = null;

      const createPopup = () => {
        if (card.querySelector(".card-preview-popup")) {
          return;
        }

        const popup = document.createElement("div");
        popup.className = "card-preview-popup";
        popup.setAttribute("aria-hidden", "true");
        popup.innerHTML = `
          <figure class="card-preview-popup__figure">
            <img src="${imagePath}" alt="Preview for ${cardLabel}">
          </figure>
        `;

        popup.style.position = "absolute";
        popup.style.left = "50%";
        popup.style.bottom = "95%";
        popup.style.transform = "translateX(-50%)";
        popup.style.opacity = "0";
        popup.style.transition = "opacity .18s ease, bottom .18s ease";
        popup.style.zIndex = "30";
        popup.style.pointerEvents = "none";

        card.appendChild(popup);

        requestAnimationFrame(() => {
          popup.style.opacity = "1";
          popup.style.bottom = "105%";
        });
      };

      const removePopup = () => {
        const popup = card.querySelector(".card-preview-popup");
        if (!popup) {
          return;
        }

        popup.style.opacity = "0";
        popup.style.bottom = "95%";
        window.setTimeout(() => {
          if (popup.isConnected) {
            popup.remove();
          }
        }, 180);
      };

      const scheduleHide = () => {
        if (removalTimer) {
          window.clearTimeout(removalTimer);
        }
        removalTimer = window.setTimeout(removePopup, 25);
      };

      const showPopup = () => {
        if (removalTimer) {
          window.clearTimeout(removalTimer);
          removalTimer = null;
        }
        createPopup();
      };

      card.addEventListener("mouseenter", showPopup);
      card.addEventListener("mouseleave", scheduleHide);
      card.addEventListener("focusin", showPopup);
      card.addEventListener("focusout", scheduleHide);
    });
  }

  function classifyAdmonitions() {
    const titles = qsAll(".admonition > .admonition-title");
    titles.forEach((title) => {
      const box = title.parentElement;
      const normalized = normalizeText(
        title.textContent || title.innerText || "",
      );

      if (/^pra(c)?tical example(s)?$/.test(normalized)) {
        box.classList.add("practical-examples");
        box.setAttribute(
          "data-badge",
          /examples$/.test(normalized) ? "EXAMPLES" : "EXAMPLE",
        );
        return;
      }

      if (/^best practice(s)?$/.test(normalized)) {
        box.classList.add("best-practice");
        box.setAttribute(
          "data-badge",
          /practices$/.test(normalized)
            ? "BEST PRACTICES"
            : "BEST PRACTICE",
        );
        return;
      }

      if (/^plot anatomy(?:$|[^a-z0-9_].*)/.test(normalized)) {
        box.classList.add("plot-anatomy");
        if (!box.hasAttribute("data-badge")) {
          box.setAttribute("data-badge", "KEY");
        }
        return;
      }

      if (/^geoprior note(?:s)?$/.test(normalized)) {
        box.classList.add("geoprior-note");
        box.setAttribute(
          "data-badge",
          /notes$/.test(normalized) ? "GEOPRIOR NOTES" : "GEOPRIOR NOTE",
        );
      }
    });
  }

  function annotateExternalLinks() {
    qsAll("main a.reference.external").forEach((link) => {
      if (link.querySelector(".gp-external-icon")) {
        return;
      }
      const icon = document.createElement("span");
      icon.className = "gp-external-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = " ↗";
      link.appendChild(icon);
    });
  }

  function enhanceSearchPlaceholder() {
    const inputs = qsAll(
      'input[type="search"], input[name="q"], .bd-search input',
    );
    inputs.forEach((input) => {
      if (!input.getAttribute("placeholder")) {
        input.setAttribute(
          "placeholder",
          "Search GeoPrior models, physics, CLI, and configs...",
        );
      }
    });
  }

  onReady(() => {
    showRecentBadges();
    setupCardPreviews();
    classifyAdmonitions();
    annotateExternalLinks();
    enhanceSearchPlaceholder();
    console.log("GeoPrior docs custom script loaded.");
  });
})();