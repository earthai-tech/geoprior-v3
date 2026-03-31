(function () {
  "use strict";

  const DEBUG = false;

  function log(...args) {
    if (DEBUG && window.console) {
      console.log("[GeoPrior]", ...args);
    }
  }

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

  function uniqueElements(items) {
    return Array.from(new Set(items.filter(Boolean)));
  }

  function normalizeText(text) {
    return String(text || "")
      .trim()
      .replace(/\s+/g, " ")
      .toLowerCase();
  }

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function getUrlRoot() {
    if (
      typeof DOCUMENTATION_OPTIONS !== "undefined" &&
      DOCUMENTATION_OPTIONS &&
      DOCUMENTATION_OPTIONS.URL_ROOT
    ) {
      return DOCUMENTATION_OPTIONS.URL_ROOT;
    }

    const html = document.documentElement;
    if (html && html.dataset && html.dataset.content_root) {
      return html.dataset.content_root;
    }

    const body = document.body;
    if (body && body.dataset && body.dataset.content_root) {
      return body.dataset.content_root;
    }

    return "./";
  }

  function buildStaticPath(relPath) {
    const cleanRel = String(relPath || "").replace(/^\/+/, "");
    if (!cleanRel) {
      return null;
    }

    const root = getUrlRoot();
    const base = new URL(root, window.location.href);
    return new URL(`_static/${cleanRel}`, base).toString();
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

  function getPreviewRegistry() {
    return {
      "card--uncertainty": {
        file: "previews/uncertainty.png",
        caption: "Forecast uncertainty overview.",
      },
      "card--errors": {
        file: "previews/errors.png",
        caption: "Error-oriented summary figure.",
      },
      "card--evaluation": {
        file: "previews/evaluation.png",
        caption: "Evaluation summary figure.",
      },
      "card--importance": {
        file: "previews/importance.png",
        caption: "Feature-importance style summary.",
      },
      "card--relationship": {
        file: "previews/relationship.png",
        caption: "Relationship-oriented analysis view.",
      },
      "card--physics": {
        file: "previews/physics.png",
        caption: "Physics-guided modeling preview.",
      },
      "card--workflow": {
        file: "previews/workflow.png",
        caption:
          "GeoPrior application workflows: staged forecasting, tuning, calibration, uncertainty handling, and reproducibility.",
      },
      "card--cli": {
        file: "previews/cli.png",
        caption: "Command-line workflow preview.",
      },
      "card--configuration": {
        file: "previews/configuration.png",
        caption: "Configuration-system preview.",
      },
      "card--core-ablation": {
        file: "previews/core_ablation.png",
        caption:
          "Core predictive performance and physics ablation: full versus no-physics comparison across accuracy, coverage, and sharpness.",
      },
      "card--ridge-bounds": {
        file: "previews/ridge_bounds.png",
        caption:
          "Bounds versus ridge summary: bound hits, ridge residuals, and clipping-versus-ridge failure modes across SM3 realizations.",
      },
      "card--hotspot-analytics": {
        file: "previews/where-to-act.png",
        caption:
          "Hotspot analytics: anomaly, exceedance probability, hotspot evolution, ranked priority clusters, and persistence for deciding where to act first.",
      },
    };
  }

  function resolvePreviewMeta(el) {
    if (!el) {
      return null;
    }

    const explicitPreview = el.dataset.preview;
    const explicitCaption =
      el.dataset.previewLabel || el.dataset.previewCaption || "";

    if (explicitPreview) {
      return {
        file: explicitPreview,
        caption: explicitCaption || getCardTitle(el),
      };
    }

    const registry = getPreviewRegistry();

    for (const [className, meta] of Object.entries(registry)) {
      if (el.classList && el.classList.contains(className)) {
        return meta;
      }
    }

    return null;
  }

  function getCardTitle(el) {
    if (!el) {
      return "GeoPrior preview";
    }

    const title =
      el.querySelector(".sd-card-title") ||
      el.querySelector(".sd-card-header") ||
      el.querySelector("strong") ||
      el.querySelector("h1, h2, h3, h4, h5, h6");

    if (!title) {
      return "GeoPrior preview";
    }

    const text = title.textContent.trim();
    return text || "GeoPrior preview";
  }

  function ensurePositioned(el) {
    if (!el) {
      return;
    }

    if (window.getComputedStyle(el).position === "static") {
      el.style.position = "relative";
    }
  }

  function ensurePopupAnchor(anchor) {
    if (!anchor) {
      return;
    }

    ensurePositioned(anchor);

    const overflow = window.getComputedStyle(anchor).overflow;
    if (overflow === "hidden") {
      anchor.style.overflow = "visible";
    }
  }

  function findMetaSourceFor(node) {
    if (!node) {
      return null;
    }

    const probes = uniqueElements([
      node,
      node.closest(".sd-card"),
      node.closest(".seealso-card"),
      node.querySelector?.(".sd-card"),
      node.querySelector?.(".seealso-card"),
    ]);

    for (const probe of probes) {
      if (probe && resolvePreviewMeta(probe)) {
        return probe;
      }
    }

    return null;
  }

  function findPreviewContexts() {
    const candidates = uniqueElements(
      qsAll(
        [
          ".see-also-tiles .seealso-card",
          ".see-also-tiles .sd-card",
          ".see-also-tiles [data-preview]",
          ".see-also-tiles [class*='card--']",
        ].join(", "),
      ),
    );

    const contexts = [];
    const seenTargets = new Set();

    candidates.forEach((node) => {
      const metaSource = findMetaSourceFor(node);
      if (!metaSource) {
        return;
      }

      const target =
        (node.classList && node.classList.contains("sd-card") && node) ||
        node.querySelector?.(".sd-card") ||
        node.closest?.(".sd-card") ||
        metaSource;

      const anchor =
        node.closest?.(".seealso-card") ||
        (node.classList && node.classList.contains("seealso-card") && node) ||
        metaSource.closest?.(".seealso-card") ||
        target;

      if (!target || seenTargets.has(target)) {
        return;
      }

      seenTargets.add(target);

      contexts.push({
        target,
        anchor: anchor || target,
        metaSource,
      });
    });

    return contexts.filter((ctx) => Boolean(resolvePreviewMeta(ctx.metaSource)));
  }

  function preloadPreviewImages(contexts) {
    const seen = new Set();

    contexts.forEach((ctx) => {
      const meta = resolvePreviewMeta(ctx.metaSource);
      if (!meta || !meta.file) {
        return;
      }

      const src = buildStaticPath(meta.file);
      if (!src || seen.has(src)) {
        return;
      }

      seen.add(src);
      const img = new Image();
      img.src = src;
    });
  }

  function removeExistingPopup(anchor) {
    if (!anchor) {
      return;
    }

    const popup = anchor.querySelector(":scope > .card-preview-popup");
    if (!popup) {
      return;
    }

    popup.classList.remove("is-visible");

    window.setTimeout(() => {
      if (popup.isConnected) {
        popup.remove();
      }
    }, 180);
  }

  function createPopup(ctx) {
    if (!ctx || !ctx.anchor || !ctx.metaSource) {
      return;
    }

    const meta = resolvePreviewMeta(ctx.metaSource);
    if (!meta || !meta.file) {
      return;
    }

    removeExistingPopup(ctx.anchor);

    const src = buildStaticPath(meta.file);
    if (!src) {
      return;
    }

    const title =
      getCardTitle(ctx.target) ||
      getCardTitle(ctx.metaSource) ||
      "GeoPrior preview";

    const caption = meta.caption || title;

    const popup = document.createElement("div");
    popup.className = "card-preview-popup";
    popup.setAttribute("aria-hidden", "true");

    popup.innerHTML = `
      <figure class="card-preview-popup__figure">
        <img
          class="card-preview-popup__image"
          src="${escapeHtml(src)}"
          alt="${escapeHtml(title)}"
          loading="lazy"
          decoding="async"
        >
        <figcaption class="card-preview-popup__caption">
          ${escapeHtml(caption)}
        </figcaption>
      </figure>
    `;

    const img = popup.querySelector(".card-preview-popup__image");
    if (img) {
      img.addEventListener("error", () => {
        log("Preview image failed to load:", src);
      });
    }

    ctx.anchor.appendChild(popup);

    requestAnimationFrame(() => {
      popup.classList.add("is-visible");
    });
  }

  function setupCardPreviews() {
    const contexts = findPreviewContexts();
    if (!contexts.length) {
      log("No preview contexts found.");
      return;
    }

    preloadPreviewImages(contexts);

    contexts.forEach((ctx) => {
      ensurePopupAnchor(ctx.anchor);

      let hideTimer = null;

      const showPopup = () => {
        if (hideTimer) {
          window.clearTimeout(hideTimer);
          hideTimer = null;
        }

        createPopup(ctx);
      };

      const scheduleHide = () => {
        if (hideTimer) {
          window.clearTimeout(hideTimer);
        }

        hideTimer = window.setTimeout(() => {
          removeExistingPopup(ctx.anchor);
        }, 50);
      };

      ctx.target.addEventListener("mouseenter", showPopup);
      ctx.target.addEventListener("mouseleave", scheduleHide);

      ctx.target.addEventListener("focusin", showPopup);
      ctx.target.addEventListener("focusout", (event) => {
        if (event.relatedTarget && ctx.target.contains(event.relatedTarget)) {
          return;
        }
        scheduleHide();
      });
    });

    log(
      "Preview contexts:",
      contexts.map((ctx) => ({
        title: getCardTitle(ctx.target),
        anchor: ctx.anchor.className,
        target: ctx.target.className,
        metaSource: ctx.metaSource.className,
      })),
    );
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
          /practices$/.test(normalized) ? "BEST PRACTICES" : "BEST PRACTICE",
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