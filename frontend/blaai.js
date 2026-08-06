const form = document.getElementById('ask-form');
const q = document.getElementById('q');
const send = document.getElementById('send');
const answer = document.getElementById('answer');

let inFlight = false;

function setBusy(busy){
    inFlight = busy;
    send.disabled = busy;
    send.textContent = busy ? '...' : 'Ask';
}

function renderSources(urls){
    if (!urls || !urls.length) return;

    const wrap = document.createElement('div');
    wrap.className = 'sources';

    const heading = document.createElement('p');
    heading.className = 'sources-heading';
    heading.textContent = urls.length === 1 ? 'Source' : 'Sources';
    wrap.appendChild(heading);

    const list = document.createElement('ul');
    for (const url of urls) {
        const item = document.createElement('li');
        const link = document.createElement('a');
        link.href = url;
        link.textContent = url.replace(/^https?:\/\/(www\.)?/, '');
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        item.appendChild(link);
        list.appendChild(item);
    }
    wrap.appendChild(list);
    answer.appendChild(wrap);
}

async function askQuestion(){
    const question = q.value.trim();
    if (!question) return;

    setBusy(true);
    answer.textContent = '';

    const body = document.createElement('div');
    body.className = 'answer-text';
    answer.appendChild(body);

    //replaced by the first token, or by the error message
    const spinner = document.createElement('img');
    spinner.className = 'loading';
    spinner.src = '/loading.gif';
    spinner.alt = 'Searching…';
    body.appendChild(spinner);

    try {
        const res = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        //proxy-level failure responses
        const contentType = res.headers.get('content-type') || '';
        if (!contentType.includes('json')) {
            throw new Error(
                res.status === 429 ? 'Too many questions. Give it a minute.'
                : res.status === 504 ? 'That took too long. Try a shorter question.'
                : `Service unavailable (${res.status}).`
            );
        }

        //errors raised before response
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            const detail = typeof err.detail === 'string' ? err.detail : null;
            throw new Error(detail || `Request failed (${res.status}).`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let started = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            //{stream: true}: a multi-byte character can straddle two chunks
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;

                let msg;
                try {
                    msg = JSON.parse(line);
                } catch {
                    continue;
                }

                if (msg.type === 'token') {
                    if (!started) { body.textContent = ''; started = true; }
                    body.textContent += msg.text;
                } else if (msg.type === 'error') {
                    const note = document.createElement('p');
                    note.className = 'error';
                    note.textContent = msg.message;
                    answer.appendChild(note);
                } else if (msg.type === 'done') {
                    renderSources(msg.sources);
                }
            }
        }
    } catch (e) {
        body.textContent = e.message;
        body.classList.add('error');
    } finally {
        setBusy(false);
    }
}

form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (inFlight) return;
    askQuestion();
});
