document.addEventListener("DOMContentLoaded", function() {
  // 获取所有语言切换链接
  const langLinks = document.querySelectorAll('a[lang]');
  
  langLinks.forEach(link => {
    // 获取当前页面路径（如 "/en/api/"）
    const currentPath = window.location.pathname;
    
    // 判断当前是英文还是中文
    if (currentPath.includes('/en/')) {
      // 如果当前是英文，点击中文时跳转到对应中文路径
      if (link.getAttribute('lang') === 'zh') {
        link.href = currentPath.replace('/en/', '/zh-cn/');
      }
    } else if (currentPath.includes('/zh-cn/')) {
      // 如果当前是中文，点击英文时跳转到对应英文路径
      if (link.getAttribute('lang') === 'en') {
        link.href = currentPath.replace('/zh-cn/', '/en/');
      }
    }
  });
});