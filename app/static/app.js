const healthPill = document.getElementById('health-pill');
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('image-file');
const resultPanel = document.getElementById('result-panel');
const previewImage = document.getElementById('preview-image');
const mainLabel = document.getElementById('main-label');
const mainScore = document.getElementById('main-score');
const modelMode = document.getElementById('model-mode');
const detectionsEl = document.getElementById('detections');
const annotatedWrapper = document.getElementById('annotated-wrapper');
const annotatedImage = document.getElementById('annotated-image');
const historyList = document.getElementById('history-list');
const refreshHistoryBtn = document.getElementById('refresh-history');

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    healthPill.textContent = data.status === 'ok' ? '服务运行正常' : '服务状态异常';
    healthPill.style.background = data.status === 'ok' ? '#dcfce7' : '#fee2e2';
    healthPill.style.color = data.status === 'ok' ? '#166534' : '#991b1b';
  } catch (err) {
    healthPill.textContent = '服务不可达';
    healthPill.style.background = '#fee2e2';
    healthPill.style.color = '#991b1b';
  }
}

function renderDetections(items = []) {
  detectionsEl.innerHTML = '';
  if (!items.length) {
    detectionsEl.innerHTML = '<div class="detection-item">未获取到可用检测目标。</div>';
    return;
  }
  items.forEach(item => {
    const el = document.createElement('div');
    el.className = 'detection-item';
    const score = Number(item.score || 0).toFixed(3);
    const raw = item.raw_label ? `（原始标签：${item.raw_label}）` : '';
    el.innerHTML = `<strong>${item.label || item.class_name}</strong><div>置信度：${score} ${raw}</div>`;
    detectionsEl.appendChild(el);
  });
}

function renderHistory(items = []) {
  historyList.innerHTML = '';
  if (!items.length) {
    historyList.innerHTML = '<div class="history-item">暂无历史记录。</div>';
    return;
  }
  items.forEach(item => {
    const el = document.createElement('article');
    el.className = 'history-item';
    const img = item.annotated_url || item.image_url;
    el.innerHTML = `
      <img class="history-thumb" src="${img}" alt="${item.filename}" />
      <h3>${item.predicted_label}</h3>
      <p>文件：${item.filename}</p>
      <p>置信度：${Number(item.confidence || 0).toFixed(3)}</p>
      <p>模型：${item.model_mode}</p>
      <p>时间：${item.created_at}</p>
    `;
    historyList.appendChild(el);
  });
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history?limit=12');
    const data = await res.json();
    renderHistory(data.items || []);
  } catch (err) {
    historyList.innerHTML = '<div class="history-item">读取历史记录失败。</div>';
  }
}

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files || !fileInput.files[0]) {
    alert('请先选择图片');
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);
  previewImage.src = URL.createObjectURL(file);

  const submitBtn = uploadForm.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  submitBtn.textContent = '识别中…';

  try {
    const res = await fetch('/api/classify', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.detail || '识别失败');
    resultPanel.classList.remove('hidden');
    mainLabel.textContent = data.predicted_label;
    mainScore.textContent = Number(data.confidence).toFixed(3);
    modelMode.textContent = `推理模式：${data.model_mode}`;
    if (data.annotated_url) {
      annotatedWrapper.classList.remove('hidden');
      annotatedImage.src = data.annotated_url + `?t=${Date.now()}`;
    } else {
      annotatedWrapper.classList.add('hidden');
      annotatedImage.src = '';
    }
    renderDetections(data.detections || []);
    await loadHistory();
  } catch (err) {
    alert(err.message || '识别失败，请稍后重试');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = '开始识别';
  }
});

refreshHistoryBtn.addEventListener('click', loadHistory);
checkHealth();
loadHistory();
