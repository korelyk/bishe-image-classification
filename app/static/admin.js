const statsEl = document.getElementById('admin-stats');
const recentEl = document.getElementById('admin-recent');
const refreshBtn = document.getElementById('admin-refresh');
const passwordInput = document.getElementById('admin-password');

function authHeaders() {
  const pwd = passwordInput.value.trim();
  return pwd ? { 'X-Admin-Password': pwd } : {};
}

function modeLabel(mode) {
  const mapping = {
    'mobilenetv2-onnx-imagenet-mapping': 'ONNX Runtime + MobileNetV2 标签映射',
    'fasterrcnn-coco': 'Faster R-CNN 目标检测',
    'resnet50-imagenet-fallback': 'ResNet50 回退分类',
  };
  return mapping[mode] || mode || '-';
}

function renderStats(stats = {}) {
  const byClass = (stats.by_class || []).map(item => `${item.predicted_label}：${item.count}`).join(' / ') || '暂无数据';
  statsEl.innerHTML = `
    <div class="stat-card"><h3>总识别次数</h3><p>${stats.total_predictions || 0}</p></div>
    <div class="stat-card"><h3>最近识别时间</h3><p style="font-size:16px;line-height:1.6">${stats.latest_prediction_at || '暂无'}</p></div>
    <div class="stat-card"><h3>类别分布</h3><p style="font-size:16px;line-height:1.6">${byClass}</p></div>
  `;
}

function renderRecent(items = []) {
  recentEl.innerHTML = '';
  if (!items.length) {
    recentEl.innerHTML = '<div class="history-item">暂无记录。</div>';
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
    recentEl.appendChild(el);
  });
}

async function loadAdmin() {
  try {
    const res = await fetch('/api/admin/stats', { headers: authHeaders() });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.detail || '加载失败');
    renderStats(data.stats || {});
    renderRecent(data.recent || []);
  } catch (err) {
    statsEl.innerHTML = `<div class="history-item">${err.message}</div>`;
    recentEl.innerHTML = '';
  }
}

refreshBtn.addEventListener('click', loadAdmin);
passwordInput.addEventListener('change', loadAdmin);
loadAdmin();
