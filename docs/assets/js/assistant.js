// assistant.js

document.addEventListener('DOMContentLoaded', function() {
    // 格式化显示 markdown
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    script.type = 'module';
    document.head.appendChild(script);
    
    var floatingIcon = document.createElement('div');
    floatingIcon.id = 'floating-icon';
    floatingIcon.innerHTML = '<img src="https://img.icons8.com/?size=100&id=121346&format=png&color=000000" alt="icon">';
    document.body.appendChild(floatingIcon);

    var dialogBox = document.createElement('div');
    dialogBox.id = 'dialog-box';
    dialogBox.innerHTML = '<div id="dialog" class="dialog"> <div id="closeButton" class="close">❌</div> <h2>LazyLLM Assistant</h2> <div id="messages" class="messages"></div> <div class="input-container"> <input type="text" id="userInput" placeholder="Your question here ..."> <button id="sendMessage">Send</button> </div> </div>';

    document.body.appendChild(dialogBox);

    const sendMessageButton = document.getElementById('sendMessage');
    const closeDialogButton = document.getElementById('closeButton');
    const userInputBox = document.getElementById('userInput');
    const messages = document.getElementById('messages');

    function changeDialobBoxDisplay() {
        if (dialogBox.style.display === 'none') {
            dialogBox.style.display = 'block';
        } else {
            dialogBox.style.display = 'none';
        }
    }

    // 加载用户消息
    function loadAndConcatUserMessages() {
        var arr = sessionStorage.getItem('lazyllmChatHistory');
        if (arr) {
            return arr;
        } else {
            return "";
        }
    }

    // 保存用户消息
    function saveUserMessages() {
        let userMessages = Array.from(messages.querySelectorAll('.user-message')).map(div => div.innerText);
        let messageStorage = userMessages.join(";")
        sessionStorage.setItem('lazyllmChatHistory', JSON.stringify(messageStorage));
    }

    // 重置用户消息
    function clearUserMessages() {
        sessionStorage.removeItem('lazyllmChatHistory');
    }
    // 发送消息的函数
    async function sendMessage() {
        const text = userInput.value.trim();
        if (text) {
            // 显示用户消息
            const userMessage = document.createElement('div');
            userMessage.className = 'message user-message';
            userMessage.innerHTML = marked.parse(text);;
            messages.appendChild(userMessage);

            // 清空输入框
            userInput.value = '';
            
            // 显示加载状态或发送请求到后端接口
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot-message';
            botMessage.textContent = '正在获取回复...';
            messages.appendChild(botMessage);
            messages.scrollTop = messages.scrollHeight; // 滚动到最新消息

            const historyArr = loadAndConcatUserMessages();

            try {
                // 发送请求到后端接口
                const response = await fetch('https://chat.lazyllm.ai/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: text }),
                });

                // 处理响应
                const data = await response.json();
                console.log(data.text)
                botMessage.innerHTML = marked.parse(data.reply || '未能获取回复');
            } catch (error) {
                botMessage.textContent = '请求失败: ' + error.message;
            }

            messages.scrollTop = messages.scrollHeight; // 滚动到最新消息
            saveUserMessages();
        }
    }

    sendMessageButton.addEventListener('click', sendMessage);
    closeDialogButton.addEventListener('click', changeDialobBoxDisplay);
    floatingIcon.addEventListener('click', changeDialobBoxDisplay);
    userInputBox.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

});
