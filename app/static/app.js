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

function renderTip(data) {
  resultTip.classList.remove('hidden');
  if (data.predicted_class === 'unknown') {
    resultTip.className = 'tip-box warning-box';
    resultTip.textContent = '当前图片未稳定映射到七类道路目标。答辩时建议优先使用首页推荐样例或典型交通场景图片。';
    return;
  }
  resultTip.className = 'tip-box success-box';
  resultTip.textContent = `本次识别已成功映射到道路目标类别：${data.predicted_label}。`;
}

function renderDetections(items = []) {
  detectionsEl.innerHTML = '';
  if (!items.length) {
    detectionsEl.innerHTML = '<div class="detection-item">未获取到可用候选标签。</div>';
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
    submitBtn.textContent = '开始识别';
  }
});

refreshHistoryBtn.addEventListener('click', loadHistory);
checkHealth();
loadHistory();
