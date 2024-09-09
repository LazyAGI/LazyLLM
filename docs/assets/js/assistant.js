document.addEventListener('DOMContentLoaded', function() {
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/showdown/dist/showdown.min.js';
    document.head.appendChild(script);
    
    var script = document.createElement('script');
    script.src = 'https://docs.lazyllm.ai/en/latest/assets/js/jigsaw.js';
    document.head.appendChild(script);
    // 对话助手入口
    var floatingIcon = document.createElement('div');
    floatingIcon.id = 'floating-icon';
    floatingIcon.innerHTML = '<img src="https://docs.lazyllm.ai/en/latest/assets/logo.png" alt="icon">';
    document.body.appendChild(floatingIcon);
    // 对话框
    var placeholdervalue = 'What is LazyLLM ?';
    var dialogBox = document.createElement('div');
    dialogBox.id = 'dialog-box';
    dialogBox.style = 'display: none';
    dialogBox.innerHTML = `
        <div id="dialog" class="dialog"> 
            <div id="reset-dialog" class="reset"></div>
            <div id="close-dialog-box" class="close"></div>
            <h2>LazyLLM Assistant</h2> 
        </div>  
        <div id="messages" class="messages"> </div> 
        <div class="input-container"> 
            <input type="text" id="user-input" placeholder="What is LazyLLM ?"> 
            <button id="send-message">Send</button> 
        </div>`;

    document.body.appendChild(dialogBox);
    // 人机验证框
    var captchaBox = document.createElement('div');
    captchaBox.id = 'captcha-box';
    captchaBox.style = "display: none";
    captchaBox.innerHTML = `
    <div id="close-captcha" class="close-captcha"></div>
    <div id="captcha-container" class="container">
        <div id="captcha"></div>
    </div>
    `;
    document.body.appendChild(captchaBox);

    const sendMessageButton = document.getElementById('send-message');
    const resetDialogButton = document.getElementById('reset-dialog');
    const closeDialogButton = document.getElementById('close-dialog-box');
    const userInput = document.getElementById('user-input');
    const messages = document.getElementById('messages');
    const closeCaptchaButton = document.getElementById('close-captcha');

    let userQuery = '';
    var isChecked = false;
    var converter;

    function setIsCheckedT() {
        isChecked = true;
    }

    function setIsCheckedF() {
        isChecked = false;
    }

    function popHelloMessage() {
        let hello = document.createElement('div');
        hello.innerHTML = `
                <div class="message received">  
                    <img src="https://docs.lazyllm.ai/en/latest/assets/logo.png" class="avatar"> 
                    <p>Hello, how can I assist you today ?</p> 
                </div> `
        messages.appendChild(hello);
    }

    function cleanToken() {
        sessionStorage.removeItem('jigsaw_token');
        setIsCheckedF();
    }

    function switchDialobBoxDisplay() {
        dialogBox.style.display = dialogBox.style.display==='none' ? 'block' : 'none';
        if (dialogBox.style.display === 'none') {
            captchaBox.style.display = 'none';
        } else {
            setTimeout(popHelloMessage, 1000);
        }
    }

    function resetDialog(){
        messages.innerHTML = '';
        sendMessageButton.disabled = false;
        setTimeout(popHelloMessage, 1000);
        cleanToken();
    }

    async function Verify() {
        if (isChecked){
            getChatContent();
        } else {
            let seed = Math.ceil(Math.random()*10000);
            try {
                const response = await fetch('https://api.lazyllm.ai/authorize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ key: seed }),
                });
            
                const data = await response.json();
                sessionStorage.setItem('jigsaw_token', data.token);
                setIsCheckedT();
                getChatContent();
            } catch (error) {
            
            }
        }
    }

    function showCaptcha() {
        captchaBox.style.display = 'block';
        if (document.getElementById('captcha').childElementCount==0){
            converter = new showdown.Converter();
            window.jigsaw.init({
                el: document.getElementById('captcha'),
                onSuccess: function() {
                    captchaBox.style.display = 'none';
                    Verify();
                    setTimeout(setIsCheckedF, 300000);
                    this.reset();
                },
                onFail: cleanToken,
                onRefresh: cleanToken,
            })
        }
    }

    function closeCaptcha() {
        captchaBox.style.display = 'none';
        userInput.value = userQuery;
        messages.removeChild(messages.lastChild);
    }

    async function getChatContent() {
        const botMessage = document.createElement('div');
        botMessage.classList.add('message');
        botMessage.classList.add('received');
        botMessage.innerHTML = `<img src="assets/logo.png" alt="机器人头像" class="avatar">
                                <div class="content">
                                    <div id="loader"></div>
                                </div>`;
        messages.appendChild(botMessage);
        messages.scrollTop = messages.scrollHeight; // 滚动到最新消息

        try {
            // 发送请求到后端接口
            const response = await fetch('https://api.lazyllm.ai/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: userQuery, token: sessionStorage.getItem('jigsaw_token') }),
            });

            // 处理响应
            const data = await response.json();
            if (data.errmsg == 200){
                botMessage.children[1].innerHTML = converter.makeHtml(data.reply);
            } else if (data.errmsg == 101){
                botMessage.children[1].innerHTML = converter.makeHtml("未能获取回复");
                cleanToken();
                closeCaptcha();
            } else {
                botMessage.children[1].innerHTML = converter.makeHtml("未能获取回复");
                botMessage.children[1].innerHTML = converter.makeHtml(data.reply);
            }
                
            userQuery = '';
            sendMessageButton.disabled = false;
            // 显示加载状态或发送请求到后端接口
        } catch (error) {
            botMessage.children[1].innerHTML = converter.makeHtml('请求失败');
        }

        messages.scrollTop = messages.scrollHeight; // 滚动到最新消息
    }
    
    function sendMessage() {
        const text = userInput.value.trim();
        if (!text) {
            userInput.value = placeholdervalue;
        }
        if (text) {
            const userMessage = document.createElement('div');
            userMessage.classList.add('message');
            userMessage.classList.add('sent');
            userMessage.innerHTML = `
                <hr/>
                <p>${text}</p>
            `;
            messages.appendChild(userMessage);

            // 清空输入框
            userQuery = text;
            userInput.value = '';
            sendMessageButton.disabled = true;

            if (!isChecked){
                showCaptcha();
            } else {
                getChatContent();
            }
        }
    }

    sendMessageButton.addEventListener('click', sendMessage);
    resetDialogButton.addEventListener('click', resetDialog);
    closeDialogButton.addEventListener('click', switchDialobBoxDisplay);
    closeCaptchaButton.addEventListener('click', closeCaptcha);
    floatingIcon.addEventListener('click', switchDialobBoxDisplay);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
