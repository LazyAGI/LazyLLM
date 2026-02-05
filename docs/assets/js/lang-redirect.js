document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");

  const routeMap = {
    'en': 'en',
    'zh': 'zh-cn'
  };

  // 使用 pathname，浏览器通常会返回 URL 编码过的路径（或者包含原始字符）
  // 稳妥起见，我们直接处理字符串，不手动编解码
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
             // 已经在当前语言页面
        } else {
            // 仅替换语言前缀，保留后续路径的所有细节（包括编码）
            const newPath = currentPath.replace(`/${currentSegment}/`, `/${targetSegment}/`);
            const finalUrl = window.location.origin + newPath + window.location.search + window.location.hash;
            
            // 1. 更新 href，保证右键菜单和 SEO 正常
            link.href = finalUrl;
            
            // 2. 强制绑定点击事件，防止 MkDocs Material 或 ReadTheDocs 脚本劫持跳转
            link.addEventListener('click', function(e) {
                // 阻止浏览器的默认链接跳转
                e.preventDefault();
                // 阻止事件冒泡和其他同类事件监听器（关键步骤，防止其他脚本干扰）
                e.stopImmediatePropagation();
                
                console.log(`[i18n] Force redirect to: ${finalUrl}`);
                window.location.href = finalUrl;
            });
        }
    }
  });
});
