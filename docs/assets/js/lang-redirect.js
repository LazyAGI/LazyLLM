document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");

  const currentUrl = new URL(window.location.href);
  const currentPath = currentUrl.pathname;
  const currentHash = currentUrl.hash;  // #xxxx
  const currentSearch = currentUrl.search; // ?key=value

  const currentLang = currentPath.startsWith('/zh-cn/') ? 'zh' : 
                     currentPath.startsWith('/en/') ? 'en' : 
                     'default';

  document.querySelectorAll('a[lang], a[hreflang]').forEach(link => {
    const targetLang = link.getAttribute('lang') || link.getAttribute('hreflang');

    if (targetLang === currentLang) {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        console.log(`[i18n] Blocked redundant switch to ${targetLang}`);
      });
      return;
    }

    const newUrl = new URL(link.href, window.location.origin);

    if (currentPath.startsWith('/zh-cn/')) {
      newUrl.pathname = currentPath.replace('/zh-cn/', '/en/');
    } 
    else if (currentPath.startsWith('/en/')) {
      newUrl.pathname = currentPath.replace('/en/', '/zh-cn/');
    }
    else {
      newUrl.pathname = targetLang === 'en' ? '/en/' : '/zh-cn/';
    }

    newUrl.search = currentSearch;
    newUrl.hash = currentHash;

    link.href = newUrl.toString();
    console.log(`[i18n] Converted to: ${newUrl}`);
  });
});
