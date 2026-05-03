"""Single-page HTML UI served at root /. Calls /v1/research/start, polls, renders markdown."""

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>aisearch — deep research</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root { color-scheme: dark; }
body { background: #0b0b0b; color: #e8e8e8; font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
       max-width: 920px; margin: 0 auto; padding: 24px; line-height: 1.55; }
h1 { font-weight: 700; letter-spacing: -0.02em; margin: 0 0 24px; }
h1 span { color: #00f7d2; font-weight: 800; }
input, select, button, textarea { font: inherit; padding: 10px 12px; background: #1a1a1a;
       color: #fff; border: 1px solid #333; border-radius: 6px; }
textarea { width: 100%; min-height: 80px; box-sizing: border-box; resize: vertical; }
.row { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.row > * { flex: 0 0 auto; }
.row > textarea { flex: 1 1 100%; }
button { background: #00f7d2; color: #000; font-weight: 700; cursor: pointer; border: none; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.status { margin-top: 24px; padding: 12px 16px; background: #141414; border-left: 3px solid #00f7d2;
          border-radius: 4px; font-size: 14px; color: #aaa; }
.status.complete { border-color: #2ecc71; }
.status.error { border-color: #e74c3c; color: #e74c3c; }
#report { margin-top: 24px; padding: 24px; background: #111; border-radius: 8px; }
#report h1, #report h2, #report h3 { margin: 1.4em 0 0.4em; color: #fff; }
#report h1 { font-size: 24px; border-bottom: 1px solid #333; padding-bottom: 8px; }
#report h2 { font-size: 18px; color: #00f7d2; }
#report a { color: #6ec1ff; text-decoration: none; }
#report a:hover { text-decoration: underline; }
#report code { background: #1a1a1a; padding: 2px 6px; border-radius: 3px; font-family: ui-monospace, monospace; }
#report ul { padding-left: 1.2em; }
#report li { margin: 4px 0; }
#report em { color: #888; font-size: 13px; }
.meta { color: #666; font-size: 12px; margin-top: 8px; font-family: ui-monospace, monospace; }
.examples { margin-top: 12px; font-size: 13px; color: #888; }
.examples a { color: #6ec1ff; text-decoration: none; cursor: pointer; }
</style>
</head>
<body>
<h1>aisearch <span>deep research</span></h1>
<form id="f">
  <div class="row">
    <textarea id="q" placeholder="Ask anything. Verifiable citations + counter-arguments + persistence."></textarea>
  </div>
  <div class="row">
    <select id="mode" title="research mode">
      <option value="general">general</option>
      <option value="competitive">competitive</option>
      <option value="academic">academic</option>
      <option value="financial">financial</option>
      <option value="legal">legal</option>
      <option value="medical">medical</option>
      <option value="geo">geo</option>
      <option value="trading">trading</option>
      <option value="people">people</option>
      <option value="product">product</option>
    </select>
    <select id="tier" title="quality tier">
      <option value="free">free (fast)</option>
      <option value="premium">premium (+adversarial)</option>
      <option value="ultra">ultra (+self-critique)</option>
    </select>
    <input type="number" id="num" value="6" min="3" max="20" style="width:80px" title="max sources">
    <button type="submit" id="go">Research</button>
  </div>
  <div class="examples">
    Try:
    <a onclick="set('How does Sticklight compete with Lovable, Framer, Webflow in 2026?','competitive')">Sticklight vs competitors</a> ·
    <a onclick="set('What are the most-cited recent papers on retrieval-augmented generation?','academic')">RAG academic</a> ·
    <a onclick="set('What is OpenAI working on right now per their latest filings + news?','financial')">OpenAI filings</a>
  </div>
</form>
<div id="status" class="status" style="display:none"></div>
<div id="report" style="display:none"></div>

<script>
function set(q, mode) { document.getElementById('q').value = q; document.getElementById('mode').value = mode; }

const f = document.getElementById('f');
const status = document.getElementById('status');
const report = document.getElementById('report');
const go = document.getElementById('go');

let polling = null;
let startTs = 0;

function statusMsg(msg, kind='') {
  status.style.display = 'block';
  status.className = 'status ' + kind;
  status.textContent = msg;
}

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const mode = document.getElementById('mode').value;
  const tier = document.getElementById('tier').value;
  const num = parseInt(document.getElementById('num').value, 10) || 6;
  go.disabled = true;
  report.style.display = 'none';
  startTs = Date.now();
  statusMsg('Starting research...');
  try {
    const r = await fetch('/v1/research/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ q, mode, tier, max_sources: num, max_sub_questions: 3, save: true }),
    });
    const data = await r.json();
    if (!data.job_id) throw new Error('no job_id');
    statusMsg(`Job ${data.job_id} running...`);
    poll(data.job_id);
  } catch (e) {
    statusMsg('Error: ' + e.message, 'error');
    go.disabled = false;
  }
});

async function poll(jobId) {
  if (polling) clearInterval(polling);
  let transientFails = 0;
  polling = setInterval(async () => {
    const elapsed = ((Date.now() - startTs) / 1000).toFixed(0);
    try {
      const r = await fetch('/v1/research/' + jobId);
      const ct = r.headers.get('content-type') || '';
      if (!r.ok || !ct.includes('application/json')) {
        transientFails++;
        if (transientFails >= 5) {
          clearInterval(polling);
          statusMsg(`Poll error: ${r.status} ${r.statusText} (gave up after 5 retries)`, 'error');
          go.disabled = false;
        } else {
          statusMsg(`Researching... ${elapsed}s elapsed (proxy hiccup ${transientFails}/5, retrying)`);
        }
        return;
      }
      transientFails = 0;
      const data = await r.json();
      if (data.status === 'complete') {
        clearInterval(polling);
        statusMsg(`Complete in ${elapsed}s · ${data.report?.sources_read}/${data.report?.sources_searched} sources · ${data.report?.claims?.length} claims`, 'complete');
        showReport(data.markdown || '', data);
        go.disabled = false;
      } else if (data.status === 'error') {
        clearInterval(polling);
        statusMsg('Error: ' + (data.error || 'unknown'), 'error');
        go.disabled = false;
      } else {
        statusMsg(`Researching... ${elapsed}s elapsed`);
      }
    } catch (e) {
      transientFails++;
      if (transientFails >= 5) {
        clearInterval(polling);
        statusMsg('Poll error: ' + e.message + ' (gave up after 5 retries)', 'error');
        go.disabled = false;
      } else {
        statusMsg(`Researching... ${elapsed}s elapsed (network blip ${transientFails}/5, retrying)`);
      }
    }
  }, 8000);
}

function showReport(md, full) {
  report.style.display = 'block';
  // very small markdown -> html (headings, links, lists)
  let html = md
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\\n## (.*)/g, '<h2>$1</h2>')
    .replace(/\\n# (.*)/g, '<h1>$1</h1>')
    .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/_([^_]+)_/g, '<em>$1</em>')
    .replace(/\\[(\\d+)\\]\\s*([^\\n]+?)\\s*\\u2014\\s*(https?:\\/\\/\\S+)/g, '<li><a href="$3" target="_blank">[$1] $2</a></li>')
    .replace(/\\n- (.+)/g, '<li>$1</li>')
    .replace(/\\n\\n/g, '<br><br>');
  report.innerHTML = html;
}
</script>
</body>
</html>
"""
