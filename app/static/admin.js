const statsEl = document.getElementById('admin-stats');
const recentEl = document.getElementById('admin-recent');
const refreshBtn = document.getElementById('admin-refresh');
const passwordInput = document.getElementById('admin-password');
const loginBtn = document.getElementById('admin-login');
const authStatus = document.getElementById('admin-auth-status');

const ADMIN_PASSWORD_STORAGE_KEY = 'bishe-admin-password';

const savedPassword = sessionStorage.getItem(ADMIN_PASSWORD_STORAGE_KEY);
if (savedPassword) {
  passwordInput.value = savedPassword;
}

function authHeaders() {
  const pwd = passwordInput.value.trim();
  return pwd ? { 'X-Admin-Password': pwd } : {};
}

function setAuthStatus(message, type = 'muted') {
  if (!authStatus) return;
  authStatus.className = `admin-hint auth-status auth-status-${type}`;
  authStatus.textContent = message;
}

function modelDisplayName(modelName) {
  const mapping = {
    resnet50: 'ResNet50',
    mobilenet_v2: 'MobileNetV2',
    efficientnet_b0: 'EfficientNet-B0',
  };
  return mapping[modelName] || modelName || '-';
}

function modeLabel(mode) {
  if (!mode) return '-';

  if (mode.startsWith('trained-road7-checkpoint:')) {
    const modelName = mode.split(':')[1];
    return `${modelDisplayName(modelName)} + 7 类微调权重`;
  }

  if (mode.startsWith('hybrid-road7-checkpoint+imagenet:')) {
    const modelName = mode.split(':')[1];
    return `${modelDisplayName(modelName)} + 微调权重/先验融合`;
  }

  if (mode.startsWith('torchvision-imagenet-fallback:')) {
    const modelName = mode.split(':')[1];
    return `${modelDisplayName(modelName)} + ImageNet 回退映射`;
  }

  const mapping = {
    'mobilenetv2-onnx-imagenet-mapping': 'MobileNetV2 + ONNX Runtime + ImageNet 映射',
    'fasterrcnn-coco': 'Faster R-CNN 目标检测',
    'resnet50-imagenet-fallback': 'ResNet50 + ImageNet 回退分类',
  };
  return mapping[mode] || mode;
}

function renderStats(stats = {}) {
  const byClass = (stats.by_class || []).map((item) => `${item.predicted_label}：${item.count}`).join(' / ') || '暂无数据';
  statsEl.innerHTML = `
    <div class="stat-card"><h3>总识别次数</h3><p>${stats.total_predictions || 0}</p></div>
    <div class="stat-card"><h3>最近识别时间</h3><p style="font-size:16px;line-height:1.6">${stats.latest_prediction_at || '暂无'}</p></div>
    <div class="stat-card"><h3>类别分布</h3><p style="font-size:16px;line-height:1.6">${byClass}</p></div>
  `;
}

async function deleteRecord(recordId, filename) {
  const confirmed = window.confirm(`确定删除记录 #${recordId}（${filename}）吗？\n这会同时删除数据库记录及对应图片文件。`);
  if (!confirmed) return;

  try {
    const res = await fetch(`/api/admin/predictions/${recordId}`, {
      method: 'DELETE',
      headers: authHeaders(),
    });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.detail || '删除失败');
    await loadAdmin();
  } catch (err) {
    alert(err.message || '删除失败，请稍后重试');
  }
}

function renderRecent(items = []) {
  recentEl.innerHTML = '';
  if (!items.length) {
    recentEl.innerHTML = '<div class="history-item">暂无记录。</div>';
    return;
  }

  items.forEach((item) => {
    const el = document.createElement('article');
    el.className = 'history-item';
    const img = item.annotated_url || item.image_url;
    el.innerHTML = `
      <img class="history-thumb" src="${img}" alt="${item.filename}" />
      <h3>${item.predicted_label}</h3>
      <p>ID：${item.id}</p>
      <p>文件：${item.filename}</p>
      <p>置信度：${Number(item.confidence || 0).toFixed(3)}</p>
      <p>模型：${modeLabel(item.model_mode)}</p>
      <p>时间：${item.created_at}</p>
      <div class="history-actions">
        <button class="btn btn-danger btn-block" type="button" data-delete-id="${item.id}">删除此记录</button>
      </div>
    `;
    const btn = el.querySelector('[data-delete-id]');
    btn.addEventListener('click', () => deleteRecord(item.id, item.filename));
    recentEl.appendChild(el);
  });
}

async function loadAdmin() {
  const pwd = passwordInput.value.trim();
  if (!pwd) {
    setAuthStatus('请输入后台登录键后点击“登录后台”。', 'warning');
    statsEl.innerHTML = '<div class="history-item">等待管理员验证。</div>';
    recentEl.innerHTML = '';
    return;
  }

  setAuthStatus('正在验证并加载后台数据...', 'muted');
  if (loginBtn) loginBtn.disabled = true;
  if (refreshBtn) refreshBtn.disabled = true;

  try {
    const res = await fetch('/api/admin/stats', { headers: authHeaders() });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.detail || '加载失败');
    sessionStorage.setItem(ADMIN_PASSWORD_STORAGE_KEY, pwd);
    setAuthStatus('验证成功，后台数据已加载。', 'success');
    renderStats(data.stats || {});
    renderRecent(data.recent || []);
  } catch (err) {
    setAuthStatus(err.message || '验证失败，请检查登录键。', 'error');
    statsEl.innerHTML = `<div class="history-item">${err.message}</div>`;
    recentEl.innerHTML = '';
  } finally {
    if (loginBtn) loginBtn.disabled = false;
    if (refreshBtn) refreshBtn.disabled = false;
  }
}

refreshBtn.addEventListener('click', loadAdmin);
if (loginBtn) {
  loginBtn.addEventListener('click', loadAdmin);
}
passwordInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    loadAdmin();
  }
});

if (savedPassword) {
  loadAdmin();
} else {
  setAuthStatus('输入登录键后点击“登录后台”，或直接按回车。', 'muted');
  statsEl.innerHTML = '<div class="history-item">等待管理员验证。</div>';
}
