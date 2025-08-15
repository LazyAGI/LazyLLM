document.addEventListener("DOMContentLoaded", function() {
  console.log("[i18n] Language redirect initialized");
  
  // 获取当前页面语言（从地址栏直接分析）
  const currentLang = window.location.pathname.startsWith('/zh-cn/') ? 'zh' : 
                     window.location.pathname.startsWith('/en/') ? 'en' : 
                     'default';
  
  // 找到所有语言切换按钮（兼容Material主题）
  document.querySelectorAll('a[lang], a[hreflang]').forEach(link => {
    // 获取按钮目标语言
    const targetLang = link.getAttribute('lang') || link.getAttribute('hreflang');
    
    // 如果当前语言与目标语言相同，则禁用跳转
    if (targetLang === currentLang) {
      link.addEventListener('click', (e) => e.preventDefault());
      console.log(`[i18n] Blocked redundant switch to ${targetLang}`);
      return;
    }
    
    // 智能路径转换
    const newPath = calculateNewPath(targetLang);
    link.href = newPath;
    console.log(`[i18n] Converted ${targetLang} link to: ${newPath}`);
  });
  
  // 路径计算函数
  function calculateNewPath(targetLang) {
    const currentPath = window.location.pathname;
    let newPath;
    
    // 当前是中文路径 → 转换到英文
    if (currentPath.startsWith('/zh-cn/')) {
      newPath = targetLang === 'en' ? 
                currentPath.replace('/zh-cn/', '/en/') : 
                currentPath; // 已经是中文则不变
    }
    // 当前是英文路径 → 转换到中文
    else if (currentPath.startsWith('/en/')) {
      newPath = targetLang === 'zh' ? 
                currentPath.replace('/en/', '/zh-cn/') : 
                currentPath; // 已经是英文则不变
    }
    // 默认路径（如首页）
    else {
      newPath = targetLang === 'en' ? '/en/' : '/zh-cn/';
    }
    
    // 保留查询参数和锚点
    return newPath + window.location.search + window.location.hash;
  }
});