const healthPill = document.getElementById('health-pill');
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('image-file');
const resultPanel = document.getElementById('result-panel');
const resultTip = document.getElementById('result-tip');
const previewImage = document.getElementById('preview-image');
const mainLabel = document.getElementById('main-label');
const mainScore = document.getElementById('main-score');
const modelMode = document.getElementById('model-mode');
const detectionsEl = document.getElementById('detections');
const annotatedWrapper = document.getElementById('annotated-wrapper');
const annotatedImage = document.getElementById('annotated-image');
const historyList = document.getElementById('history-list');
const refreshHistoryBtn = document.getElementById('refresh-history');

function modeLabel(mode) {
  const mapping = {
    'mobilenetv2-onnx-imagenet-mapping': 'ONNX Runtime + MobileNetV2 标签映射',
    'fasterrcnn-coco': 'Faster R-CNN 目标检测',
    'resnet50-imagenet-fallback': 'ResNet50 回退分类',
  };
  return mapping[mode] || mode || '-';
}

function formatTime(isoString) {
  if (!isoString) return '-';
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return isoString;
  return date.toLocaleString('zh-CN', { hour12: false });
}

function setHealthUI(ok, timeText, labelText) {
  healthPill.classList.remove('status-ok', 'status-bad');
  healthPill.classList.add(ok ? 'status-ok' : 'status-bad');
  healthPill.innerHTML = `
    <span class="health-dot"></span>
    <span class="health-text">${labelText}</span>
    <span class="health-time">${timeText}</span>
  `;
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const ok = data.status === 'ok';
    setHealthUI(ok, `更新于 ${formatTime(data.timestamp)}`, ok ? '系统运行正常' : '系统状态异常');
  } catch (err) {
    setHealthUI(false, '健康检查未通过', '系统暂不可达');
  }
}

function renderTip(data) {
  resultTip.classList.remove('hidden');
  if (data.predicted_class === 'unknown') {
    resultTip.className = 'tip-box warning-box';
    resultTip.textContent = '当前图片未能稳定映射到预设的七类道路目标，建议更换典型道路交通场景图片后重新测试。';
    return;
  }
  resultTip.className = 'tip-box success-box';
  resultTip.textContent = `本次识别结果已成功映射到道路目标类别：${data.predicted_label}。`;
}

function renderDetections(items = []) {
  detectionsEl.innerHTML = '';
  if (!items.length) {
    detectionsEl.innerHTML = '<div class="detection-item">当前未获取到可用候选结果。</div>';
    return;
  }
  items.forEach((item, index) => {
    const el = document.createElement('div');
    el.className = 'detection-item';
    const score = Number(item.score || 0).toFixed(3);
    const raw = item.raw_label ? `原始标签：${item.raw_label}` : '原始标签：-';
    const mapped = item.class_name === 'unknown' ? '未映射到七类' : `映射结果：${item.label || item.class_name}`;
    el.innerHTML = `<strong>Top-${index + 1}</strong><div>${raw}</div><div>${mapped}</div><div>置信度：${score}</div>`;
    detectionsEl.appendChild(el);
  });
}

function renderHistory(items = []) {
  historyList.innerHTML = '';
  if (!items.length) {
    historyList.innerHTML = '<div class="history-item">当前暂无识别记录。</div>';
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
      <p>模型：${modeLabel(item.model_mode)}</p>
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
    historyList.innerHTML = '<div class="history-item">识别记录读取失败，请稍后重试。</div>';
  }
}

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files || !fileInput.files[0]) {
    alert('请先选择待识别图片');
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);
  previewImage.src = URL.createObjectURL(file);

  const submitBtn = uploadForm.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  submitBtn.textContent = '识别处理中…';

  try {
    const res = await fetch('/api/classify', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.detail || '识别失败');
    resultPanel.classList.remove('hidden');
    mainLabel.textContent = data.predicted_label;
    mainScore.textContent = Number(data.confidence).toFixed(3);
    modelMode.textContent = `推理模式：${modeLabel(data.model_mode)}`;
    renderTip(data);
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
    submitBtn.textContent = '提交图片并开始识别';
  }
});

refreshHistoryBtn.addEventListener('click', loadHistory);
checkHealth();
loadHistory();
