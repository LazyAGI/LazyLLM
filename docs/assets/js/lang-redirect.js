document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");

  const routeMap = {
    'en': 'en',
    'zh': 'zh-cn'
  };

  // Use pathname; browsers typically return URL-encoded paths (or raw characters)
  const currentPath = window.location.pathname;
  
  console.log(`[i18n] Current Raw Pathname: ${currentPath}`);

  document.querySelectorAll('a[lang], a[hreflang]').forEach(link => {
    const targetLang = link.getAttribute('lang') || link.getAttribute('hreflang');
    
    if (!targetLang || !routeMap[targetLang]) return;

    const targetSegment = routeMap[targetLang];
    
    let currentSegment = null;
    for (const segment of Object.values(routeMap)) {
      if (currentPath.startsWith(`/${segment}/`)) {
        currentSegment = segment;
        break;
      }
    }

    if (currentSegment) {
        if (currentSegment === targetSegment) {
             // Already on the target language page: update href to the current full URL
             // This makes clicking the button refresh the current page instead of jumping to the default homepage configured in mkdocs.yml
             const currentUrl = window.location.href;
             link.href = currentUrl;

             // Also add forced redirect logic to prevent theme or other scripts from interfering
             link.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopImmediatePropagation();
                console.log(`[i18n] Reloading current page: ${currentUrl}`);
                window.location.href = currentUrl;
             });
        } else {
            // Replace only the language prefix, preserving the rest of the path exactly (including encoding)
            const newPath = currentPath.replace(`/${currentSegment}/`, `/${targetSegment}/`);
            const finalUrl = window.location.origin + newPath + window.location.search + window.location.hash;
            
            // 1. Update href so right-click menu and SEO remain correct
            link.href = finalUrl;
            
            // 2. Force-bind a click event to prevent other scripts from hijacking the redirect
            link.addEventListener('click', function(e) {
                // Prevent browser's default link navigation
                e.preventDefault();
                // Stop propagation and other similar event listeners (critical to prevent interference from other scripts)
                e.stopImmediatePropagation();
                
                console.log(`[i18n] Force redirect to: ${finalUrl}`);
                window.location.href = finalUrl;
            });
        }
    }
  });
});
