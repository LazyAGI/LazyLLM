document.addEventListener("DOMContentLoaded", () => {
  // 动态加载 Lottie Player 脚本
  var script = document.createElement('script');
  script.src = 'https://unpkg.com/@dotlottie/player-component@latest/dist/dotlottie-player.mjs';
  script.type = 'module';
  document.head.appendChild(script);

  // 创建一个半透明背景层
  var overlay = document.createElement('div');
  overlay.id = 'overlay';
  overlay.style.display = 'none';  // 初始状态隐藏
  overlay.style.position = 'fixed';
  overlay.style.top = 0;
  overlay.style.left = 0;
  overlay.style.width = '100%';
  overlay.style.height = '100%';
  overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  overlay.style.zIndex = 9998;
  document.body.appendChild(overlay);

  // 创建一个 Streamlit iframe
  var iframeContainer = document.createElement('div');
  iframeContainer.id = 'iframe-container';
  iframeContainer.style.display = 'none';  // 初始状态隐藏
  iframeContainer.style.position = 'fixed';
  iframeContainer.style.top = '50%';
  iframeContainer.style.left = '50%';
  iframeContainer.style.width = '800px';
  iframeContainer.style.height = '600px';
  iframeContainer.style.border = '1px solid #ccc';
  iframeContainer.style.borderRadius = '10px';
  iframeContainer.style.transform = 'translate(-50%, -50%)';
  iframeContainer.style.backgroundColor = '#ffffff'; // 背景颜色
  iframeContainer.style.zIndex = 9999;
  document.body.appendChild(iframeContainer);

  var iframe = document.createElement('iframe');
  iframe.src = 'http://localhost:8501';
  iframe.id = 'streamlit-iframe';
  iframe.style.width = '100%';
  iframe.style.height = '100%';
  iframe.style.border = 'none';
  iframeContainer.appendChild(iframe);

  // 创建一个 Lottie 动画容器
  var loadingAnimation = document.createElement('dotlottie-player');
  loadingAnimation.src = 'https://lottie.host/f14abc98-e72d-476d-a559-794b9e08f320/8FqLciivbW.json';
  loadingAnimation.style.width = '300px';
  loadingAnimation.style.height = '300px';
  loadingAnimation.style.position = 'absolute';
  loadingAnimation.style.top = '50%';
  loadingAnimation.style.left = '50%';
  loadingAnimation.style.transform = 'translate(-50%, -50%)';
  loadingAnimation.style.display = 'none'; // 初始状态隐藏
  loadingAnimation.background = 'transparent';
  loadingAnimation.speed = '1';
  loadingAnimation.loop = true;
  loadingAnimation.autoplay = true;
  iframeContainer.appendChild(loadingAnimation);

  // 创建一个机器人按钮
  var button = document.createElement('button');
  button.id = 'chatbot-button';
  button.textContent = '🐈';
  button.style.position = 'fixed';
  button.style.bottom = '60px';
  button.style.right = '60px';
  button.style.width = '80px';
  button.style.height = '80px';
  button.style.backgroundColor = '#010810';
  button.style.color = '#ffffff';
  button.style.border = 'none';
  button.style.borderRadius = '50%';
  button.style.fontSize = '40px';
  button.style.display = 'flex';
  button.style.alignItems = 'center';
  button.style.justifyContent = 'center';
  button.style.cursor = 'pointer';
  button.style.zIndex = 10000;
  document.body.appendChild(button);

  // 点击按钮时切换 iframeContainer 和 overlay 的显示/隐藏
  button.addEventListener('click', function(event) {
    event.stopPropagation();  // 阻止事件冒泡
    if (iframeContainer.style.display === 'none') {
      iframeContainer.style.display = 'block';
      overlay.style.display = 'block';
    } else {
      iframeContainer.style.display = 'none';
      overlay.style.display = 'none';
    }
  });

  // 点击 overlay 时隐藏 iframeContainer 和 overlay
  overlay.addEventListener('click', function() {
    iframeContainer.style.display = 'none';
    overlay.style.display = 'none';
  });

  // 处理 iframe 加载错误
  iframe.addEventListener('load', function() {
    try {
      var iframeDocument = iframe.contentDocument || iframe.contentWindow.document;
      if (iframeDocument.title === '404 Not Found' || iframeDocument.title === '403 Forbidden') {
        // 如果检测到错误页面
        iframe.style.display = 'none';
        loadingAnimation.style.display = 'block';
      } else {
        // 正常加载
        iframe.style.display = 'block';
        loadingAnimation.style.display = 'none';
      }
    } catch (e) {
      // 跨域问题可能会阻止访问 iframe 内容
      console.error('Error accessing iframe content:', e);
      iframe.style.display = 'none';
      loadingAnimation.style.display = 'block';
    }
  });
  
  iframe.addEventListener('error', function() {
    // 处理 iframe 加载错误
    iframe.style.display = 'none';
    loadingAnimation.style.display = 'block';
  });

  // 处理 iframe 加载成功
  iframe.addEventListener('load', function() {
    iframe.style.display = 'block';
    loadingAnimation.style.display = 'none';
  });
});