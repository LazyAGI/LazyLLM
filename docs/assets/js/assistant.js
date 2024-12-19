document.addEventListener('DOMContentLoaded', function() {
    var script = document.createElement('script');
    script.src = 'https://api.lazyllm.top/static/scripts/chatAssistant.js'
    // 确保在外部脚本加载完成后执行后续逻辑
    script.onload = function() {
        var logopath = "../../assets/logo.png";
        let pathName = window.document.location.pathname;  
        let paths = pathName.replace(/^\/|\/$/g, '').split('/');
        if (paths.length == 2){
            logopath = "assets/logo.png";
        } 
        
        // 对话助手入口
        const chatAssistant = new ChatAssistant({
            logopath: logopath,
            apiBaseUrl: 'https://api.lazyllm.top',
            language: paths[0],
            version: paths[1],
            placeholder: 'Type your question here...',
        });
    };

    // 将脚本添加到文档中
    document.head.appendChild(script);
});