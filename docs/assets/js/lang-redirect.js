document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");

  const routeMap = {
    'en': 'en',
    'zh-cn': 'zh-cn'
  };

  const currentPath = decodeURIComponent(window.location.pathname);

  document.querySelectorAll('a[lang], a[hreflang]').forEach(link => {
    const targetLang = link.getAttribute('lang') || link.getAttribute('hreflang');
    const targetSegment = routeMap[targetLang];

    if (!targetSegment) return;

    let processed = false;
    for (const [langCode, segment] of Object.entries(routeMap)) {
      if (currentPath.startsWith(`/${segment}/`)) {
        if (segment === targetSegment) {
          // Already on target language, prevent click
          link.addEventListener('click', (e) => {
            e.preventDefault();
            console.log(`[i18n] Already on language: ${targetLang}`);
          });
        } else {
          // Calculate new path
          const newPath = currentPath.replace(`/${segment}/`, `/${targetSegment}/`);
          const newUrl = new URL(newPath, window.location.origin);
          newUrl.search = window.location.search;
          newUrl.hash = window.location.hash;
          const finalUrl = newUrl.toString();

          // 1. Update href attribute (for hover and SEO)
          link.href = finalUrl;

          // 2. Force click event binding to ensure redirect
          link.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent default behavior and interference from other scripts
            console.log(`[i18n] Force redirect to: ${finalUrl}`);
            window.location.href = finalUrl;
          });
          
          console.log(`[i18n] Setup redirect for ${targetLang} to: ${finalUrl}`);
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
