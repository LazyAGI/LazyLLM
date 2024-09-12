document.addEventListener('DOMContentLoaded', function() {
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/showdown/dist/showdown.min.js'
    document.head.appendChild(script);
    
    var script = document.createElement('script');
    script.src = 'assets/js/jigsaw.js'
    document.head.appendChild(script);
    // 对话助手入口
    var floatingIcon = document.createElement('div');
    floatingIcon.id = 'floating-icon';
    floatingIcon.innerHTML = '<img src="https://docs.lazyllm.ai/en/latest/assets/logo.png" alt="icon">';
    document.body.appendChild(floatingIcon);
    // 对话框
    var placeholdervalue = 'What is LazyLLM ?'
    var dialogBox = document.createElement('div');
    dialogBox.id = 'dialog-box';
    dialogBox.style = 'display: none'
    dialogBox.innerHTML = `
    <div class="dialog-container">
        <div id="dialog" class="dialog"> 
            <div id="reset-dialog" class="reset"></div>
            <div id="close-dialog-box" class="close"></div>
            <h2>LazyLLM Assistant</h2> 
        </div>  
        <div id="messages" class="messages"> </div> 
        <div class="input-container"> 
            <input type="text" id="user-input" placeholder="What is LazyLLM ?"> 
            <button id="send-message">Send</button> 
        </div>
    </div>`

    document.body.appendChild(dialogBox);
    // 人机验证框
    var captchaBox = document.createElement('div');
    captchaBox.id = 'captcha-box';
    captchaBox.style = "display: none"
    captchaBox.innerHTML = `
    <div id="close-captcha" class="close-captcha"></div>
    <div id="captcha-container" class="container">
        <div id="captcha"></div>
    </div>
    `
    document.body.appendChild(captchaBox);

    const sendMessageButton = document.getElementById('send-message');
    const resetDialogButton = document.getElementById('reset-dialog');
    const closeDialogButton = document.getElementById('close-dialog-box');
    const userInput = document.getElementById('user-input');
    const messages = document.getElementById('messages');
    const closeCaptchaButton = document.getElementById('close-captcha');

    var userQuery = '';
    var isChecked = false;
    var isWaiting = false;
    var firstHello = true;
    var showSource = true;
    var converter;
    
    function setIsCheckedT() {
        isChecked = true;
    }

    function setIsCheckedF() {
        isChecked = false;
    }

    function setMessageState(state) {
        isWaiting = state;
        sendMessageButton.disabled = state;
    }
    
    function popHelloMessage() {
        if (!isWaiting && firstHello){
            let hello = document.createElement('div');
            hello.innerHTML = `
                    <div class="message received">  
                        <img src="https://docs.lazyllm.ai/en/latest/assets/logo.png" class="avatar"> 
                        <p>Hello, how can I assist you today ?</p> 
                    </div> `
            messages.appendChild(hello)
            firstHello = false;
        }
    }

    function cleanToken() {
        sessionStorage.removeItem('jigsaw_token');
        setIsCheckedF();
    }

    function handleClickOutside(event) {
        if (event.target === dialogBox) {
            dialogBox.style.display = 'none';
            // 移除点击事件监听器
            window.removeEventListener('click', handleClickOutside);
        } 
    }

    function switchDialobBoxDisplay() {
        dialogBox.style.display = dialogBox.style.display==='none' ? 'block' : 'none';
        if (dialogBox.style.display === 'none') {
            captchaBox.style.display = 'none';
            window.removeEventListener('click', handleClickOutside);
        } else {
            window.addEventListener('click', handleClickOutside);
            setTimeout(popHelloMessage, 1000);
        }
    }

    function resetDialog(){
        messages.innerHTML = '';
        setMessageState(false);
        cleanToken();
        firstHello = true;
        setTimeout(popHelloMessage, 1000);
    }

    async function Verify() {
        if (isChecked){
            getChatContent();
        } else {
            let seed = Math.ceil(Math.random()*10000);
            try {
                const response = await fetch('http://lazyllm.top/authorize', {
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
        setMessageState(false);
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
                                </div>
                                `;
        botMessage.children[1].addEventListener('click', function(e) {
            console.log(e);
            navigator.clipboard.writeText(e.target.innerText);
        });
        messages.appendChild(botMessage);
        messages.scrollTop = messages.scrollHeight; // 滚动到最新消息

        try {
            // 发送请求到后端接口
            const response = await fetch('http://lazyllm.top/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: userQuery, token: sessionStorage.getItem('jigsaw_token') }),
            });

            // 处理响应
            const data = await response.json();
            converter = new showdown.Converter();
            if (data.errmsg == 200){
                botMessage.children[1].innerHTML = converter.makeHtml(data.reply);
                if (showSource){
                    var source_box = document.createElement('div');
                    source_box.innerHTML = converter.makeHtml(data.sources);
                    botMessage.children[1].appendChild(source_box);
                }
            } else if (data.errmsg == 102 || data.errmsg==103){
                botMessage.children[1].innerHTML = converter.makeHtml(data.reply);
            } else {
                botMessage.children[1].innerHTML = converter.makeHtml("未能获取回复，请稍后再试");
                cleanToken();
                closeCaptcha();
            }
        } catch (error) {
            botMessage.children[1].innerHTML = converter.makeHtml('请求失败');
        }
        userQuery = '';
        setMessageState(false);
        messages.scrollTop = messages.scrollHeight; // 滚动到最新消息
    }
    
    function sendMessage() {
        if (isWaiting){
            return;
        }
        const text = userInput.value.trim();
        if (!text) { userInput.value = placeholdervalue; }
        else {
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
            
            setMessageState(true);

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
