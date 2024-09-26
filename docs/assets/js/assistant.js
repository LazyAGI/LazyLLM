document.addEventListener('DOMContentLoaded', function() {
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/showdown/dist/showdown.min.js'
    document.head.appendChild(script);

    var logopath = "../../assets/logo.png";
    let pathName = window.document.location.pathname;  
    let paths = pathName.replace(/^\/|\/$/g, '').split('/');
    var versionInfo = '/' + paths.slice(0, 2).join('/');
    if (paths.length == 2){
        logopath = "assets/logo.png";
    } 
    // 对话助手入口
    var floatingIcon = document.createElement('div');
    floatingIcon.id = 'floating-icon';
    floatingIcon.innerHTML = `<img src="${logopath}" alt="icon">`;
    document.body.appendChild(floatingIcon);

    var language;
    var userQuery = '';
    var botReply = '';
    var sessionid = '';
    var isWaiting = false;
    var firstHello = true;
    var dialogInit = true;
    var converter;
    var placeholdervalue;

    var sendMessageButton;
    var resetDialogButton;
    var userInput;
    var messages;
    var dialogBox;
    var captchaBox;
    // 对话框
    async function initDialogDom() {
        let curVersion = 'stable';
        language = paths[0];
        curVersion = paths[1];

        try {
            // 发送请求到后端接口
            const response = await fetch('https://api.lazyllm.top/checkVersion', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ version: curVersion }),
            });

            // 处理响应
            const data = await response.json();
            curVersion = data.version;
            placeholdervalue = data.suggestedQuestion;
        } catch (error) {
            curVersion = 'stable';
            placeholdervalue = 'What is LazyLLM ?';
        } finally {
            let sendbtn = language=='zh-cn' ? '发送' : 'Send';
            dialogBox = document.createElement('div');
            dialogBox.id = 'dialog-box';
            dialogBox.style = 'display: block'
            dialogBox.innerHTML = `
            <div class="dialog-container">
                <div id="dialog" class="dialog"> 
                    <div class="dropdown">
                        <div class="dropdown-icon"></div>
                        <button class="dropdown-button">${curVersion}</button>
                    </div>
                    <div id="reset-dialog" class="reset"></div>
                </div>
                <div id="messages" class="messages"> </div> 
                <div class="input-container"> 
                    <input type="text" id="user-input" placeholder="${placeholdervalue}"> 
                    <button id="send-message">${sendbtn}</button> 
                </div>
            </div>`

            document.body.appendChild(dialogBox);
            dialogInit = false;
            sendMessageButton = document.getElementById('send-message');
            resetDialogButton = document.getElementById('reset-dialog')
            userInput = document.getElementById('user-input');
            messages = document.getElementById('messages');
            bindEvents();
            setTimeout(popHelloMessage, 1000);
            userInput.focus();
        }
    }
    
    // 人机验证框
    function initCaptchaDom(){
        captchaBox = document.createElement('div');
        captchaBox.id = 'captcha-box';
        captchaBox.style = "display: none";
        captchaBox.innerHTML = `
        <div id="close-captcha" class="close-captcha"></div>
        <div id="captcha-container" class="container">
            <div id="captcha"></div>
        </div>
        `
        document.body.appendChild(captchaBox);
        let closeCaptchaButton = document.getElementById('close-captcha');
        closeCaptchaButton.addEventListener('click', closeCaptcha);
    }

    async function sendFeedback(e, feedback, state) {
        let message_id = e.target.dataset.id;
        let like = state==0 ? feedback.slice(0, -7) : feedback;
        try {
            // 发送请求到后端接口
            const response = await fetch('https://api.lazyllm.top/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ type: like, state: state, message_id: message_id }),
            });

            data = await response.json();
            if (state==1) {
                let feedback_box = e.target.parentNode;
                feedback_box.children[0].className = data.stateu;
                feedback_box.children[1].className = data.staten;
            } else {
                e.target.className = like == 'useful' ? 'useful-normal' : 'notuseful-normal';
            }
        } catch (error) {
            
        }
    }

    function clickFeedback(e) {
        let state = e.target.classList[0];
        if (state.includes('normal')) {
            if (state.includes('notuseful')) {
                sendFeedback(e, 'notuseful', 1);
            } else {
                sendFeedback(e, 'useful', 1);
            }
        } else {
            if (state.includes('notuseful')) {
                sendFeedback(e, 'notuseful-normal', 0);
            } else {
                sendFeedback(e, 'useful-normal', 0);
            }
        }
    }

    function initFeedbackDom(message_id, domElement) {
        var util_box = document.createElement('div');
        util_box.classList.add('util-box');

        var likeButton = document.createElement('button');
        likeButton.className = 'useful-normal';
        likeButton.setAttribute('data-id', message_id);
        likeButton.addEventListener('click', clickFeedback);
        util_box.appendChild(likeButton);

        var dislikeButton = document.createElement('button');
        dislikeButton.className = 'notuseful-normal';
        dislikeButton.setAttribute('data-id', message_id);
        dislikeButton.addEventListener('click', clickFeedback);
        util_box.appendChild(dislikeButton);

        domElement.children[1].appendChild(util_box);
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
                        <img src="${logopath}" class="avatar"> 
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
        }
    }

    function switchDialobBoxDisplay() {
        if (dialogInit) {
            dialogInit = false;
            initDialogDom();
            initCaptchaDom();
            converter = new showdown.Converter();
            return;
        } 

        if (dialogBox.style.display === 'none') {
            dialogBox.style.display = 'block';
            window.addEventListener('click', handleClickOutside);
            setTimeout(popHelloMessage, 100);
            userInput.focus();
        } else {
            dialogBox.style.display = 'none';
            captchaBox.style.display = 'none';
            window.removeEventListener('click', handleClickOutside);
        }
    }

    function resetDialog(){
        messages.innerHTML = '';
        setMessageState(false);
        sessionid = '';
        firstHello = true;
        setTimeout(popHelloMessage, 100);
    }

    async function Verify() {
        let seed = Math.ceil(Math.random()*10000);
        try {
            const response = await fetch('https://api.lazyllm.top/authorize', {
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

    function fetchWithTimeout(url, options = {}, timeout = 120000) {
        // 创建一个新的 Promise 来处理超时逻辑
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Request timed out')), timeout)
        );
    
        // 执行 fetch 请求
        const fetchPromise = fetch(url, options);
    
        // 使用 Promise.race() 来处理 fetch 请求和超时逻辑
        return Promise.race([fetchPromise, timeoutPromise]);
    }

    function stream_response_handler(data, domElement) {
        data = data.data[0];
        if (data.state == 'success') {
            if (data.errmsg == 200) {
                if (botReply==''){
                    domElement.children[1].innerHTML = converter.makeHtml(data.message.response);
                }
                var source_box = document.createElement('div');
                source_box.classList.add('source-box');
                source_box.style = `height: ${data.message.sources.length>2 ? 80 : 40}px`
                data.message.sources.forEach(source => {
                    const link = document.createElement('a');
                    link.textContent = source.text;
                    link.href = versionInfo + source.href;
                    link.target = '_blank'
                    source_box.appendChild(link);
                });
                domElement.children[1].appendChild(source_box);
                sessionid = data.session_id;
                initFeedbackDom(data.stable_id, domElement);
            } else if (data.errmsg == 102 || data.errmsg==103){
                // 触发敏感词或无关问题, 不计入某段session
                domElement.children[1].innerHTML = converter.makeHtml(data.message.response);
                setMessageState(false);
            } else if (data.errmsg == 101) {
                // token 过期
                messages.removeChild(messages.lastChild);
                showCaptcha();
            } else {
                domElement.children[1].innerHTML = converter.makeHtml("Something went wrong, please try again later.");
                setMessageState(false);
            }
            messages.scrollTop = messages.scrollHeight; 
            return true;
        }

        botReply += data.response;
        if (data.state == 'codeblock'){
            domElement.children[1].innerHTML = converter.makeHtml(botReply + '\n```');
        } else {
            domElement.children[1].innerHTML = converter.makeHtml(botReply);
        }
        
        messages.scrollTop = messages.scrollHeight; 
        return false;
    }

    async function getChatContent() {
        const botMessage = document.createElement('div');
        botMessage.classList.add('message');
        botMessage.classList.add('received');
        botMessage.innerHTML = `<img src="${logopath}" alt="机器人头像" class="avatar">
                                <div class="content">
                                    <div id="loader"></div>
                                </div>
                                `;
        botMessage.children[1].addEventListener('click', function(e) {
            navigator.clipboard.writeText(e.target.innerText);
        });
        messages.appendChild(botMessage);
        messages.scrollTop = messages.scrollHeight; // 滚动到最新消息

        try {
            fetchWithTimeout('https://api.lazyllm.top/chatStream', { 
                method: 'POST',  
                headers: {
                    'Accept': 'text/event-stream',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: userQuery, token: sessionStorage.getItem('jigsaw_token'), sessionid: sessionid }),})
                .then(response => {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder('utf-8');
                    var resFin = false;
                    // 启动递归读取流
                    function readStream() {
                        // 读取流中的一段数据
                        return reader.read().then(result => {
                            // 解码接收到的数据块
                            const chunk = decoder.decode(result.value, { stream: true });
                            
                            // 逐行解析 JSON 数据并显示
                            chunk.split('\n').forEach(line => {
                                if (line.trim()) {  // 跳过空行
                                    // console.log(line)
                                    resFin = stream_response_handler(JSON.parse(line), botMessage)
                                }
                            });
                            if (resFin) {
                                return;
                            }
                            // 递归调用以继续读取下一段数据
                            return readStream();
                        });
                    }
                    readStream();
                    setMessageState(false);
                })
                .catch(error => {
                    // 500 404等
                    botMessage.children[1].innerHTML = converter.makeHtml("Request failed, please try again later.");
                    setMessageState(false);
                });
        } catch (error) {
            botMessage.children[1].innerHTML = converter.makeHtml("Something went wrong, please try again later");
            setMessageState(false);
        }
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
            botReply = '';
            userInput.value = '';
            
            setMessageState(true);
            getChatContent();
        }
    }

    floatingIcon.addEventListener('click', switchDialobBoxDisplay);

    function bindEvents() {
        window.addEventListener('click', handleClickOutside);
        sendMessageButton.addEventListener('click', sendMessage);
        resetDialogButton.addEventListener('click', resetDialog);
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
});
