document.addEventListener('DOMContentLoaded', function () {
    const chatForm = document.getElementById('chat-form');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
  
    function addMessage(message, sender) {
      const messageDiv = document.createElement('div');
      messageDiv.classList.add('message');
      messageDiv.textContent = `${sender}: ${message}`;
      chatMessages.appendChild(messageDiv);
    }
  
    chatForm.addEventListener('submit', function (e) {
      e.preventDefault();
      const userMessage = userInput.value.trim();
      
      if (userMessage !== '') {
        addMessage(userMessage, 'You');
        userInput.value = '';
        scrollToBottom();
        sendMessageToBackend(userMessage);
      }
    });
  
    function scrollToBottom() {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  

    function sendMessageToBackend(message) {
      fetch('/get_response', {
        method: 'POST',
        body: new URLSearchParams({ 'user_input': message }),
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      })
      .then(response => response.json())
      .then(data => {
        addMessage(data.reply, 'Chatbot');
        scrollToBottom();
      })
      .catch(error => console.error('Error:', error));
    }
  });  