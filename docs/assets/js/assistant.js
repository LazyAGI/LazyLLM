document.addEventListener('DOMContentLoaded', function() {
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/showdown/dist/showdown.min.js'
    document.head.appendChild(script);
    
    var script = document.createElement('script');
    script.src = '/assets/js/jigsaw.js'
    document.head.appendChild(script);
    // 对话助手入口
    var floatingIcon = document.createElement('div');
    floatingIcon.id = 'floating-icon';
    floatingIcon.innerHTML = '<img src="/assets/logo.png" alt="icon">';
    document.body.appendChild(floatingIcon);
    // 对话框
    var placeholdervalue = 'What is LazyLLM ?';
    
    var dialogBox = document.createElement('div');
    dialogBox.id = 'dialog-box';
    dialogBox.style = 'display: none'
    dialogBox.innerHTML = `
    <div class="dialog-container">
        <div id="dialog" class="dialog"> 
            <div class="dropdown">
                <div class="dropdown-icon"></div>
                <button class="dropdown-button">stable</button>
                <div class="dropdown-content" style="dispaly:none">
                </div>
            </div>
            <div id="reset-dialog" class="reset"></div>
            
        </div>
        <div id="messages" class="messages"> </div> 
        <div class="input-container"> 
            <input type="text" id="user-input" placeholder="${placeholdervalue}"> 
            <button id="send-message">Send</button> 
        </div>
    </div>`

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
    `
    document.body.appendChild(captchaBox);

    var copySucceed = document.createElement('div');
    copySucceed.innerHTML = `
    <span class='copy-success'>Copied to clipboard</span>
    `

    const sendMessageButton = document.getElementById('send-message');
    const resetDialogButton = document.getElementById('reset-dialog');
    const userInput = document.getElementById('user-input');
    const messages = document.getElementById('messages');
    const closeCaptchaButton = document.getElementById('close-captcha');
    
    const dropdownButton = document.querySelector('.dropdown-button');
    const dropdownContent = document.querySelector('.dropdown-content');

    var userQuery = '';
    var isWaiting = false;
    var firstHello = true;
    var currentVersion = 1002;
    var converter;

    (function addversion() {
        const versions = [{"text": "latest", "id": 1001}, {"text": "stable", "id": 1002}, {"text":"v0.2.2", "id": 1022}, {"text":"v0.2.3", "id":1023}];
        versions.forEach(version => {
            const link = document.createElement('span');
            link.textContent = version.text;
            link.addEventListener('click', function (event) {
                event.preventDefault();
                currentVersion = version.value;
                dropdownButton.textContent = `${version.text}`;
                dropdownContent.style.display = 'none';
                event.stopPropagation();
            });
            dropdownContent.appendChild(link);
        });
    })();

    function setMessageState(state) {
        isWaiting = state;
        sendMessageButton.disabled = state;
    }
    
    function popHelloMessage() {
        if (!isWaiting && firstHello){
            let hello = document.createElement('div');
            hello.innerHTML = `
                    <div class="message received">  
                        <img src="/assets/logo.png" class="avatar"> 
                        <p>Hello, how can I assist you today ?</p> 
                    </div> `
            messages.appendChild(hello)
            firstHello = false;
        }
    }

    function handleClickOutside(event) {
        if (event.target === dialogBox) {
            dialogBox.style.display = 'none';
            window.removeEventListener('click', handleClickOutside);
        } else {
            dropdownContent.style.display = 'none';
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
            userInput.focus();
        }
    }

    function resetDialog(){
        messages.innerHTML = '';
        setMessageState(false);
        firstHello = true;
        setTimeout(popHelloMessage, 1000);
    }

    async function Verify() {
        let seed = Math.ceil(Math.random()*10000);
        try {
            const response = await fetch('http://api.lazyllm.top/authorize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ key: seed }),
            });
        
            const data = await response.json();
            sessionStorage.setItem('jigsaw_token', data.token);
            getChatContent();
        } catch (error) {
        
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
                    this.reset();
                },
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
        botMessage.innerHTML = `<img src="/assets/logo.png" alt="机器人头像" class="avatar">
                                <div class="content">
                                    <div id="loader"></div>
                                </div>
                                `;
        botMessage.children[1].addEventListener('click', function(e) {
            navigator.clipboard.writeText(e.target.innerText);
            console.log(e)
        });
        messages.appendChild(botMessage);
        messages.scrollTop = messages.scrollHeight; // 滚动到最新消息

        try {
            // 发送请求到后端接口
            const response = await fetch('http://api.lazyllm.top/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: userQuery, token: sessionStorage.getItem('jigsaw_token'), version: currentVersion }),
            });

            // 处理响应
            const data = await response.json();
            converter = new showdown.Converter();
            if (data.errmsg == 200){
                botMessage.children[1].innerHTML = converter.makeHtml(data.reply);
                
                var source_box = document.createElement('div');
                data.sources.forEach(source => {
                    const link = document.createElement('a');
                    link.textContent = source.text;
                    link.href = source.href;
                    link.target = '_black'
                    source_box.appendChild(link);
                });
                botMessage.children[1].appendChild(source_box);
                setMessageState(false);
            } else if (data.errmsg == 102 || data.errmsg==103){
                // 触发敏感词或无关问题
                botMessage.children[1].innerHTML = converter.makeHtml(data.reply);
                setMessageState(false);
            } else if (data.errmsg == 101) {
                // token 过期
                messages.removeChild(messages.lastChild);
                showCaptcha();
            } else {
                botMessage.children[1].innerHTML = converter.makeHtml("未能获取回复，请稍后再试");
                setMessageState(false);
            }
        } catch (error) {
            botMessage.children[1].innerHTML = converter.makeHtml('请求失败');
            setMessageState(false);
        }
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
            userQuery = text;
            userInput.value = '';
            
            setMessageState(true);
            getChatContent();
        }
    }

    sendMessageButton.addEventListener('click', sendMessage);
    resetDialogButton.addEventListener('click', resetDialog);
    closeCaptchaButton.addEventListener('click', closeCaptcha);
    dropdownButton.addEventListener('click', function (event) {
        dropdownContent.style.display = 'block';
        event.stopPropagation();
    });
    floatingIcon.addEventListener('click', switchDialobBoxDisplay);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

});
