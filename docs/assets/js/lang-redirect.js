document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");
  
  // 获取当前完整URL组成部分
  const currentUrl = new URL(window.location.href);
  const currentPath = currentUrl.pathname;
  const currentHash = currentUrl.hash;  // #xxxx
  const currentSearch = currentUrl.search; // ?key=value

  // 分析当前语言状态
  const currentLang = currentPath.startsWith('/zh-cn/') ? 'zh' : 
                     currentPath.startsWith('/en/') ? 'en' : 
                     'default';

  // 处理所有语言切换链接
  document.querySelectorAll('a[lang], a[hreflang]').forEach(link => {
    const targetLang = link.getAttribute('lang') || link.getAttribute('hreflang');
    
    // 阻止重复语言切换
    if (targetLang === currentLang) {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        console.log(`[i18n] Blocked redundant switch to ${targetLang}`);
      });
      return;
    }

    // 计算新路径（核心改进：使用URL对象保留下所有参数）
    const newUrl = new URL(link.href, window.location.origin);
    
    // 路径转换逻辑
    if (currentPath.startsWith('/zh-cn/')) {
      newUrl.pathname = currentPath.replace('/zh-cn/', '/en/');
    } 
    else if (currentPath.startsWith('/en/')) {
      newUrl.pathname = currentPath.replace('/en/', '/zh-cn/');
    }
    // 默认首页情况
    else {
      newUrl.pathname = targetLang === 'en' ? '/en/' : '/zh-cn/';
    }

    // 保留原始参数和哈希（关键修复点）
    newUrl.search = currentSearch;
    newUrl.hash = currentHash;

    // 更新链接
    link.href = newUrl.toString();
    console.log(`[i18n] Converted to: ${newUrl}`);
  });
});