document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");

  const routeMap = {
    'en': 'en',
    'zh': 'zh-cn'
  };

  const currentPath = window.location.pathname;

  document.querySelectorAll('a[lang], a[hreflang]').forEach(link => {
    const targetLang = link.getAttribute('lang') || link.getAttribute('hreflang');
    const targetSegment = routeMap[targetLang];

    if (!targetSegment) return;

    let processed = false;
    for (const [langCode, segment] of Object.entries(routeMap)) {
      if (currentPath.startsWith(`/${segment}/`)) {
        if (segment === targetSegment) {
          link.addEventListener('click', (e) => {
            e.preventDefault();
            console.log(`[i18n] Already on language: ${targetLang}`);
          });
        } else {
          // Replace only the language segment prefix, keeping the rest of the path (version, page, etc) intact
          const newPath = currentPath.replace(`/${segment}/`, `/${targetSegment}/`);
          
          const newUrl = new URL(newPath, window.location.origin);
          newUrl.search = window.location.search;
          newUrl.hash = window.location.hash;

          link.href = newUrl.toString();
          console.log(`[i18n] Updated ${targetLang} link to: ${newUrl}`);
        }
        processed = true;
        break;
      }
    }

    if (!processed && targetSegment) {
        console.log(`[i18n] Path ${currentPath} did not match known language segments, redirect logic skipped.`);
    }
  });
});
