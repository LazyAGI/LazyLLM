document.addEventListener("DOMContentLoaded", function() {
  console.log("[Language Redirect] Script loaded");
  
  // 修改选择器：同时匹配lang和hreflang
  const langLinks = document.querySelectorAll('a[lang], a[hreflang]');
  console.log(`Found ${langLinks.length} language links`);

  langLinks.forEach(link => {
    const lang = link.getAttribute('lang') || link.getAttribute('hreflang');
    const currentPath = window.location.pathname;
    const originalHref = link.getAttribute('href');
    
    console.log(`Processing ${lang} link: ${originalHref} (current: ${currentPath})`);

    // 计算新路径
    let newHref;
    if (currentPath.startsWith('/en/')) {
      newHref = currentPath.replace('/en/', '/zh-cn/');
    } else if (currentPath.startsWith('/zh-cn/')) {
      newHref = currentPath.replace('/zh-cn/', '/en/');
    } else {
      newHref = lang === 'en' ? '/en/' : '/zh-cn/';
    }

    // 保留锚点和查询参数
    newHref += window.location.search + window.location.hash;
    
    link.setAttribute('href', newHref);
    console.log(`Rewrote link to: ${newHref}`);
  });
});