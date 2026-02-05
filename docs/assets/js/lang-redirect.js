document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");

  const routeMap = {
    'en': 'en',
    'zh': 'zh-cn'
  };

  const currentPath = decodeURIComponent(window.location.pathname);
  
  console.log(`[i18n] Current Full URL: ${window.location.href}`);
  console.log(`[i18n] Current Pathname: ${currentPath}`);

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
          console.log(`[i18n] Switching from '${segment}' to '${targetSegment}'`);
          const newPath = currentPath.replace(`/${segment}/`, `/${targetSegment}/`);
          console.log(`[i18n] Calculated New Path: ${newPath}`);
          
          const newUrl = new URL(newPath, window.location.origin);
          newUrl.search = window.location.search;
          newUrl.hash = window.location.hash;
          const finalUrl = newUrl.toString();

          // 1. Update href attribute (for hover and SEO)
          link.href = finalUrl;

          // Remove the forced click listener which might be conflicting with RtD logic
          // or causing race conditions. Let the native href navigation work.
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
