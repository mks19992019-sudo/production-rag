const launcher = document.getElementById('chat-launcher');
const widget = document.getElementById('chat-widget');
const closeButton = document.getElementById('close-chat');
const form = document.getElementById('chat-form');
const input = document.getElementById('user');
const sendButton = document.getElementById('click');
const messages = document.getElementById('messages');

const CHAT_URL = 'http://localhost:8000/chat';
// A new id is intentionally created on every page load/refresh.
const threadId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;

function toggleChat(open) {
    widget.classList.toggle('open', open);
    widget.setAttribute('aria-hidden', String(!open));
    launcher.setAttribute('aria-expanded', String(open));
    if (open) input.focus();
}

function addMessage(text, from) {
    const message = document.createElement('div');
    message.className = `message ${from}-message`;
    message.innerHTML = `<div class="bubble"></div><time>Now</time>`;
    message.querySelector('.bubble').textContent = text;
    messages.appendChild(message);
    messages.scrollTop = messages.scrollHeight;
    return message;
}

function responseText(data) {
    if (typeof data === 'string') return data;
    if (Array.isArray(data)) return data.join('\n');
    if (data && typeof data.answer === 'string') return data.answer;
    if (data && typeof data.message === 'string') return data.message;
    return JSON.stringify(data);
}

async function sendMessage() {
    const text = input.value.trim();
    if (!text || sendButton.disabled) return;

    addMessage(text, 'user');
    input.value = '';
    sendButton.disabled = true;
    input.disabled = true;
    const thinkingMessage = addMessage('Thinking...', 'assistant');

    try {
        const response = await fetch(CHAT_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ msg: text, thread_id: threadId })
        });
        const data = await response.json();
        if (!response.ok) {
            const detail = data?.detail || `Request failed (${response.status})`;
            throw new Error(Array.isArray(detail) ? detail.join(', ') : detail);
        }
        thinkingMessage.querySelector('.bubble').textContent = responseText(data);
    } catch (error) {
        console.error('Chat request failed:', error);
        thinkingMessage.querySelector('.bubble').textContent = `Could not get a response: ${error.message}`;
    } finally {
        sendButton.disabled = false;
        input.disabled = false;
        input.focus();
        messages.scrollTop = messages.scrollHeight;
    }
}

launcher.addEventListener('click', () => toggleChat(!widget.classList.contains('open')));
closeButton.addEventListener('click', () => toggleChat(false));
form.addEventListener('submit', (event) => {
    event.preventDefault();
    sendMessage();
});
