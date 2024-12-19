(function () {
    class ChatAssistant {
      constructor(options) {
        // 参数处理
        this.logopath = options.logopath || 'default-logo.png';
        this.apiBaseUrl = options.apiBaseUrl || 'https://api.lazyllm.top';
        this.language = options.language || 'en';
        this.placeholder = options.placeholder || 'How can I help you?';
        this.curVersion = options.version || 'stable';

        // 初始化状态变量
        this.isWaiting = false;
        this.firstHello = true;
        this.dialogInit = true;
        this.userQuery = '';
        this.botReply = '';
        this.sessionId = '';

        this.loadCSS(`${this.apiBaseUrl}/static/scripts/style.css`)
        this.loadJS(`${this.apiBaseUrl}/static/scripts/jigsaw.js`)
        this.loadJS("http://cdn.jsdelivr.net/npm/showdown/dist/showdown.min.js")

        // 初始化 DOM 和事件绑定
        this.initFloatingIcon();
        this.clickFeedback = this.clickFeedback.bind(this);
      }

      loadCSS(cssUrl) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = cssUrl;
        document.head.appendChild(link);
      }

      loadJS(jsUrl) {
        const script = document.createElement('script');
        script.src = jsUrl;
        script.async = true; // 异步加载
        document.head.appendChild(script);
      }

      initFloatingIcon() {
        const floatingIcon = document.createElement('div');
        floatingIcon.id = 'floating-icon';
        floatingIcon.innerHTML = `<img src="${this.logopath}" alt="icon">`;
        document.body.appendChild(floatingIcon);

        floatingIcon.addEventListener('click', () => this.switchDialogBoxDisplay());
      }

      async initDialogDom() {
        try {
          const response = await fetch(`${this.apiBaseUrl}/checkVersion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ version: this.curVersion }),
          });

          const data = await response.json();
          this.curVersion = data.version;
          this.placeholder = data.suggestedQuestion;
        } catch (error) {
          this.curVersion = 'stable';
          this.placeholder = 'What is LazyLLM?';
        } finally {
          this.renderDialogBox();
        }
      }

      initCaptchaDom(){
        const captchaBox = document.createElement('div');
        captchaBox.id = 'captcha-box';
        captchaBox.style = "display: none";
        captchaBox.innerHTML = `
        <div id="close-captcha" class="close-captcha"></div>
        <div id="captcha-container" class="container">
            <div id="captcha"></div>
        </div>
        `
        document.body.appendChild(captchaBox);
        this.captchaBox = captchaBox;
        let closeCaptchaButton = document.getElementById('close-captcha');
        closeCaptchaButton.addEventListener('click', () => this.closeCaptcha());
      }

      initFeedbackDom(message_id, domElement) {
        var util_box = document.createElement('div');
        util_box.classList.add('util-box');

        var likeButton = document.createElement('button');
        likeButton.className = 'useful-normal';
        likeButton.setAttribute('data-id', message_id);
        likeButton.addEventListener('click', this.clickFeedback);
        util_box.appendChild(likeButton);

        var dislikeButton = document.createElement('button');
        dislikeButton.className = 'notuseful-normal';
        dislikeButton.setAttribute('data-id', message_id);
        dislikeButton.addEventListener('click', this.clickFeedback);
        util_box.appendChild(dislikeButton);

        domElement.children[1].appendChild(util_box);
      }

      renderDialogBox() {
        var dialogBox = document.createElement('div');
        dialogBox.id = 'dialog-box';
        dialogBox.style.display = 'block';
        dialogBox.innerHTML = `
          <div class="dialog-container">
            <div id="dialog" class="dialog">
              <div class="dropdown">
                <div class="dropdown-icon"></div>
                <button class="dropdown-button">${this.curVersion}</button>
              </div>
              <div id="reset-dialog" class="reset"></div>
            </div>
            <div id="messages" class="messages"></div>
            <div class="input-container">
              <textarea id="user-input" placeholder="${this.placeholder}"></textarea>
              <button id="send-message">${this.language === 'zh-cn' ? '发送' : 'Send'}</button>
            </div>
          </div>`;
        document.body.appendChild(dialogBox);

        this.bindEvents(dialogBox);
        this.popHelloMessage();
      }

      bindEvents(dialogBox) {
        this.sendMessageButton = dialogBox.querySelector('#send-message');
        this.resetDialogButton = dialogBox.querySelector('#reset-dialog');
        this.userInput = dialogBox.querySelector('#user-input');
        this.messages = dialogBox.querySelector('#messages');

        this.sendMessageButton.addEventListener('click', () => this.sendMessage());
        this.resetDialogButton.addEventListener('click', () => this.resetDialog());
        this.userInput.focus();

        let that = this;
        let userInput = this.userInput;
        userInput.addEventListener('input', (e) => {
            userInput.style.height = '40px';
            userInput.style.height = e.target.scrollHeight + 'px';
        });
        userInput.addEventListener('keypress', function(e) {
            // shift + enter 换行
            if (e.shiftKey && e.key === 'Enter') {
                // 阻止默认事件
                e.preventDefault();
                // 将光标移动到换行后的正确位置
                const cursorPos = userInput.selectionStart;
                userInput.value = userInput.value.substring(0, cursorPos) + "\n" + userInput.value.substring(cursorPos);
                userInput.selectionStart = userInput.selectionEnd = cursorPos + 1;
                // 调整高度
                userInput.style.height = 40 + "px";
                userInput.style.height = userInput.scrollHeight + "px";
            } else if (e.key === 'Enter') {
                e.preventDefault();
                that.sendMessage();
            }
        });
        window.addEventListener('click', this.handleClickOutside);
      }

      fetchWithTimeout(url, options = {}, timeout = 120000) {
        // 创建一个新的 Promise 来处理超时逻辑
        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Request timed out')), timeout)
        );

        // 执行 fetch 请求
        const fetchPromise = fetch(url, options);

        // 使用 Promise.race() 来处理 fetch 请求和超时逻辑
        return Promise.race([fetchPromise, timeoutPromise]);
      }

      async sendFeedback(e, feedback, state) {
        let message_id = e.target.dataset.id;
        let like = state==0 ? feedback.slice(0, -7) : feedback;
        try {
            // 发送请求到后端接口
            const response = await fetch(`${this.apiBaseUrl}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ type: like, state: state, message_id: message_id }),
            });

            let data = await response.json();
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

      clickFeedback(e) {
        let state = e.target.classList[0];
        if (state.includes('normal')) {
            if (state.includes('notuseful')) {
              this.sendFeedback(e, 'notuseful', 1);
            } else {
              this.sendFeedback(e, 'useful', 1);
            }
        } else {
            if (state.includes('notuseful')) {
              this.sendFeedback(e, 'notuseful-normal', 0);
            } else {
              this.sendFeedback(e, 'useful-normal', 0);
            }
        }
      }

      stream_response_handler(res, domElement) {
        let converter = this.converter;
        let data = res.data[0];
        if (data.state=='success'){
          this.updateSourceInfo(domElement, data)
          this.messages.scrollTop = this.messages.scrollHeight;
          return true;
        } else if (["thinking", "planning", "solving"].some(sub => sub == data.state)){
          domElement.children[1].innerHTML = `<p class="state-word">${data.state} ...</p>`
        } else if (data.state == 'codeblock'){
          this.botReply += data.response;
          domElement.children[1].innerHTML = converter.makeHtml(this.botReply + '\n```');
        } else{
          this.botReply += data.response;
          domElement.children[1].innerHTML = converter.makeHtml(this.botReply);
        }
        this.messages.scrollTop = this.messages.scrollHeight;
        return false;
      }

      updateSourceInfo(domElement, data) {
        let converter = this.converter;
        let error_code = data.errmsg;
        if (error_code == 200){
          if (this.botReply==''){
            domElement.children[1].innerHTML = converter.makeHtml(data.message.response);
          }
          let source_box = document.createElement('div');
          source_box.classList.add('source-box');
          source_box.style = `height: ${data.message.sources.length>2 ? 80 : 40}px`
          data.message.sources.forEach(source => {
            const link = document.createElement('a');
            link.textContent = source.text;
            if (source.href.startsWith("/")) {
              link.href = this.curVersion + source.href;
            } else { link.href = source.href; }
            link.target = '_blank'
            source_box.appendChild(link);
          });
          domElement.children[1].appendChild(source_box);
          this.sessionid = data.session_id;
          this.initFeedbackDom(data.stable_id, domElement);
        }else if (error_code==102 || error_code==103){
          // 触发敏感词或无关问题, 不计入某段session
          domElement.children[1].innerHTML = converter.makeHtml(data.message.response);
          this.setMessageState(false);
        }else if (error_code==101){
          // token 过期
          this.messages.removeChild(this.messages.lastChild);
          this.showCaptcha();
        }else{
          domElement.children[1].innerHTML = converter.makeHtml("Something went wrong, please try again later.");
          this.setMessageState(false);
        }
      }

      async getChatContent() {
        let that = this;
        let messages = that.messages;
        let botMessage = document.createElement('div');
        botMessage.classList.add('message', 'received');
        botMessage.innerHTML = `<img src="${that.logopath}" alt="机器人头像" class="avatar">
                                <div class="content">
                                    <div id="loader"></div>
                                </div>`;
        botMessage.children[1].addEventListener('click', function(e) {
            navigator.clipboard.writeText(e.target.innerText);
        });
        messages.appendChild(botMessage);
        messages.scrollTop = messages.scrollHeight;

        try {
            that.fetchWithTimeout(`${this.apiBaseUrl}/chatStream`, {
                method: 'POST',
                headers: {
                    'Accept': 'text/event-stream',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: that.userQuery, token: sessionStorage.getItem('jigsaw_token'), sessionid: that.sessionid, version: that.curVersion }),})
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
                                    resFin = that.stream_response_handler(JSON.parse(line), botMessage)
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
                    that.setMessageState(false);
                })
                .catch(error => {
                    // 500 404等
                    botMessage.children[1].innerHTML = that.converter.makeHtml("Request failed, please try again later!");
                    that.setMessageState(false);
                });
        } catch (error) {
            botMessage.children[1].innerHTML = that.converter.makeHtml("Something went wrong, please try again later!");
            that.setMessageState(false);
        }
      }

      sendMessage() {
        if (this.isWaiting) return;

        let userInput = this.userInput;
        const text = userInput.value.trim();
        if (!text) { userInput.value = this.placeholder; return;}

        const userMessage = document.createElement('div');
        userMessage.classList.add('message', 'sent');
        userMessage.innerHTML = `<hr/>
                                 <p style="white-space: pre-wrap;">${text}</p>`;
        this.messages.appendChild(userMessage);

        this.userQuery = text;
        this.botReply = '';

        this.resetUserInput();
        this.setMessageState(true);
        this.getChatContent();
      }

      popHelloMessage() {
        if (!this.isWaiting && this.firstHello){
            let hello = document.createElement('div');
            hello.innerHTML = `
                    <div class="message received">
                        <img src="${this.logopath}" class="avatar">
                        <p>Hello, how can I assist you today ?</p>
                    </div> `
            this.messages.appendChild(hello)
            this.firstHello = false;
        }
      }

      handleClickOutside(event) {
        if (event.target.id == 'dialog-box') {
            let dialogBox = document.getElementById("dialog-box");
            let captchaBox = document.getElementById("captcha-box");
            dialogBox.style.display = 'none';
            captchaBox.style.display = 'none';
            window.removeEventListener('click', this.handleClickOutside);
        }
      }

      setMessageState(state) {
        this.isWaiting = state;
        this.sendMessageButton.disabled = state;
      }

      resetUserInput() {
        this.userInput.value = '';
        this.userInput.style.height = '40px';
      }

      resetDialog() {
        this.messages.innerHTML = '';
        this.userInput.value = '';
        this.sessionId = '';
        this.isWaiting = false;
        this.firstHello = true;
        this.captchaBox.style.display = 'none';
        setTimeout(() => this.popHelloMessage(), 500);
        console.log("!23")
      }

      switchDialogBoxDisplay() {
        if (this.dialogInit) {
          this.dialogInit = false;
          this.initDialogDom();
          this.initCaptchaDom();
          this.converter = new showdown.Converter();
          return;
        }

        let dialogBox = document.querySelector("#dialog-box");
        if (dialogBox.style.display === 'none') {
            dialogBox.style.display = 'block';
            window.addEventListener('click', this.handleClickOutside);
            setTimeout(() => this.popHelloMessage(), 500);
            this.userInput.focus();
        } else {
            dialogBox.style.display = 'none';
            this.captchaBox.style.display = 'none';
            window.removeEventListener('click', this.handleClickOutside);
        }
      }

      async Verify() {
        let that = this;
        let seed = Math.ceil(Math.random()*10000);
        try {
            const response = await fetch(`${this.apiBaseUrl}/authorize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ key: seed }),
            });

            const data = await response.json();
            sessionStorage.setItem('jigsaw_token', data.token);
            that.getChatContent();
        } catch (error) {

        }
      }

      showCaptcha() {
        let that =  this;
        that.captchaBox.style.display = 'block';
        if (document.getElementById('captcha').childElementCount==0){
            that.converter = new showdown.Converter();
            window.jigsaw.init({
                el: document.getElementById('captcha'),
                onSuccess: function() {
                    that.captchaBox.style.display = 'none';
                    that.Verify();
                    this.reset();
                },
            })
        }
      }

      closeCaptcha() {
        this.captchaBox.style.display = 'none';
        this.setMessageState(false);
        this.userInput.value = this.userQuery;
        this.messages.removeChild(this.messages.lastChild);
      }
    }

    // 暴露到全局
    window.ChatAssistant = ChatAssistant;
  })();
