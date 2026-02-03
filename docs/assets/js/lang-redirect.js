document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");

  const routeMap = {
    'en': 'en',
    'zh': 'zh-cn'
  };

  // Ensure we work with decoded path to avoid %20 doubling issues or mismatches
  const currentPath = decodeURIComponent(window.location.pathname);

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
          // Replace language segment in the decoded path
          const newPath = currentPath.replace(`/${segment}/`, `/${targetSegment}/`);
          
          // Construct full URL
          // new URL() will handle encoding special characters back to %20 etc. correctly
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
