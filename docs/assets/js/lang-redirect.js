document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");

  const routeMap = {
    'en': 'en',
    'zh': 'zh-cn'
  };

  const currentPath = window.location.pathname;
  const pathParts = currentPath.split('/');
  
  const currentLangSegment = pathParts[1];

  document.querySelectorAll('a[lang], a[hreflang]').forEach(link => {
    const targetLang = link.getAttribute('lang') || link.getAttribute('hreflang');
    const targetSegment = routeMap[targetLang];

    if (!targetSegment) return;

    if (targetSegment === currentLangSegment) {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        console.log(`[i18n] Blocked redundant switch to ${targetLang}`);
      });
      return;
    }

    const newPathParts = [...pathParts];
    if (newPathParts.length > 1) {
        newPathParts[1] = targetSegment;

        const newUrl = new URL(newPathParts.join('/'), window.location.origin);

        newUrl.search = window.location.search;
        newUrl.hash = window.location.hash;

        link.href = newUrl.toString();
        console.log(`[i18n] Converted to: ${newUrl}`);
    }
  });
});
