document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");

  const routeMap = {
    'en': 'en',
    'zh': 'zh-cn'
  };

  // Use pathname; browsers typically return URL-encoded paths (or raw characters)
  const currentPath = window.location.pathname;
  
  console.log(`[i18n] Current Raw Pathname: ${currentPath}`);

  function getCurrentSegment(path) {
    for (const segment of Object.values(routeMap)) {
      if (path.startsWith(`/${segment}/`)) {
        return segment;
      }
    }
    return null;
  }

  function getFromDataToLlmTargetPath(path, currentSegment, targetSegment) {
    const marker = '/Tutorial/from-data-to-llm/';
    if (targetSegment !== 'en' || !path.includes(marker)) return null;

    const markerIndex = path.indexOf(marker);
    const versionPrefix = path.substring(0, markerIndex);
    const targetPrefix = versionPrefix.replace(`/${currentSegment}/`, `/${targetSegment}/`);
    return `${targetPrefix}${marker}`;
  }

  function getTargetUrl(targetSegment, includeHash = true) {
    const freshPath = window.location.pathname;
    const currentSegment = getCurrentSegment(freshPath);
    if (!currentSegment) return null;
    const specialPath = getFromDataToLlmTargetPath(freshPath, currentSegment, targetSegment);
    const freshNewPath = specialPath || freshPath.replace(`/${currentSegment}/`, `/${targetSegment}/`);
    const hash = specialPath || !includeHash ? '' : window.location.hash;
    return window.location.origin + freshNewPath + window.location.search + hash;
  }

  document.querySelectorAll('a[lang], a[hreflang]').forEach(link => {
    const targetLang = link.getAttribute('lang') || link.getAttribute('hreflang');
    
    if (!targetLang || !routeMap[targetLang]) return;

    const targetSegment = routeMap[targetLang];
    
    let currentSegment = getCurrentSegment(currentPath);

    if (currentSegment) {
        if (currentSegment === targetSegment) {
             // Already on the target language page: update href to the current full URL
             // This makes clicking the button refresh the current page instead of jumping to
             // the default homepage configured in mkdocs.yml
             const currentUrl = window.location.href;
             link.href = currentUrl;

             // Also add forced redirect logic to prevent theme or other scripts from interfering
             link.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopImmediatePropagation();
                // Get fresh URL in case hash changed after page load
                const freshUrl = window.location.href;
                console.log(`[i18n] Reloading current page: ${freshUrl}`);
                window.location.href = freshUrl;
             });
        } else {
            // Replace only the language prefix, preserving the rest of the path exactly (including encoding)
            const specialPath = getFromDataToLlmTargetPath(currentPath, currentSegment, targetSegment);
            const newPath = specialPath || currentPath.replace(`/${currentSegment}/`, `/${targetSegment}/`);
            const finalHash = specialPath ? '' : window.location.hash;
            const finalUrl = window.location.origin + newPath + window.location.search + finalHash;
            
            // 1. Update href so right-click menu and SEO remain correct
            link.href = finalUrl;
            
            // 2. Force-bind a click event to prevent other scripts from hijacking the redirect
            link.addEventListener('click', function(e) {
                // Prevent browser's default link navigation
                e.preventDefault();
                // Stop propagation and other similar event listeners.
                // This prevents interference from other scripts.
                e.stopImmediatePropagation();
                
                // Dynamically get current path and hash (user might have navigated in-page)
                const freshPath = window.location.pathname;
                const freshHash = window.location.hash;
                const freshSearch = window.location.search;

                // Construct target base URL
                const freshCurrentSegment = getCurrentSegment(freshPath) || currentSegment;
                const specialPath = getFromDataToLlmTargetPath(freshPath, freshCurrentSegment, targetSegment);
                const freshNewPath = specialPath || freshPath.replace(`/${freshCurrentSegment}/`, `/${targetSegment}/`);
                const targetBaseUrl = window.location.origin + freshNewPath + freshSearch;

                if (freshHash && !specialPath) {
                    console.log(`[i18n] Hash detected: ${freshHash}. Verifying existence in target...`);
                    
                    fetch(targetBaseUrl)
                        .then(response => {
                            if (!response.ok) throw new Error("Target page fetch failed");
                            return response.text();
                        })
                        .then(html => {
                            const parser = new DOMParser();
                            const doc = parser.parseFromString(html, "text/html");
                            const tagId = freshHash.substring(1); // Remove '#'
                            
                            // Check if element with this ID exists in target document
                            const element = doc.getElementById(tagId);
                            
                            let finalDest = targetBaseUrl;
                            if (element) {
                                finalDest += freshHash;
                                console.log(`[i18n] Hash found in target. Keeping tag.`);
                            } else {
                                console.log(`[i18n] Hash NOT found in target. Dropping tag.`);
                            }
                            
                            window.location.href = finalDest;
                        })
                        .catch(err => {
                            console.error("[i18n] Error checking hash, fallback to direct redirect:", err);
                            // Fallback: assume it works or just redirect
                            window.location.href = targetBaseUrl + freshHash;
                        });
                } else {
                    console.log(`[i18n] Force redirect to: ${targetBaseUrl}`);
                    window.location.href = targetBaseUrl;
                }
            });
        }
    }
  });

  document.querySelectorAll('a[href^="/en/"], a[href^="/zh-cn/"]').forEach(link => {
    const href = link.getAttribute('href');
    const targetSegment = href.startsWith('/en/') ? 'en' : 'zh-cn';
    const finalUrl = getTargetUrl(targetSegment);
    if (!finalUrl) return;
    link.href = finalUrl;
  });
});
