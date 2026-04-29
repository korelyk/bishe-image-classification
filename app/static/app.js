const healthPill = document.getElementById('health-pill');

const singleUploadForm = document.getElementById('single-upload-form');
const singleFileInput = document.getElementById('single-image-file');
const singleModelSelect = document.getElementById('single-model-name');
const singleGradcamCheckbox = document.getElementById('single-with-gradcam');
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

const scenarioModelSelect = document.getElementById('scenario-model-name');
const batchForm = document.getElementById('batch-form');
const batchFilesInput = document.getElementById('batch-files');
const batchResults = document.getElementById('batch-results');
const stressForm = document.getElementById('stress-form');
const stressFileInput = document.getElementById('stress-image-file');
const stressResults = document.getElementById('stress-results');
const startCameraBtn = document.getElementById('start-camera');
const captureCameraBtn = document.getElementById('capture-camera');
const cameraVideo = document.getElementById('camera-video');
const cameraCanvas = document.getElementById('camera-canvas');

const comparisonBody = document.getElementById('comparison-table-body');
const comparisonSummary = document.getElementById('comparison-summary');
const robustnessCards = document.getElementById('robustness-cards');
const trainingSummaryBody = document.getElementById('training-summary-body');
const trainingSummaryNote = document.getElementById('training-summary-note');
const reportConclusionNote = document.getElementById('report-conclusion-note');

const overviewBestModel = document.getElementById('overview-best-model');
const overviewBestAccuracy = document.getElementById('overview-best-accuracy');
const overviewFastestModel = document.getElementById('overview-fastest-model');
const overviewFastestFps = document.getElementById('overview-fastest-fps');
const overviewDatasetSize = document.getElementById('overview-dataset-size');
const overviewTrainingTotal = document.getElementById('overview-training-total');

const reportBestModel = document.getElementById('report-best-model');
const reportBestAccuracy = document.getElementById('report-best-accuracy');
const reportFastestModel = document.getElementById('report-fastest-model');
const reportFastestFps = document.getElementById('report-fastest-fps');
const reportDatasetSize = document.getElementById('report-dataset-size');
const reportTrainingTotal = document.getElementById('report-training-total');

const sampleRunButtons = Array.from(document.querySelectorAll('[data-sample-run]'));

let currentStream = null;
let comparisonCache = null;
let trainingSummaryCache = null;
let robustnessCache = null;

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

  return mode;
}

function formatTime(isoString) {
  if (!isoString) return '-';
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return isoString;
  return date.toLocaleString('zh-CN', { hour12: false });
}

function formatDatasetSize(datasetSize) {
  if (!datasetSize || typeof datasetSize !== 'object') return '-';
  return `训练 ${datasetSize.train ?? '-'} / 验证 ${datasetSize.val ?? '-'} / 测试 ${datasetSize.test ?? '-'}`;
}

function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return '-';
  const numeric = Number(seconds);
  if (numeric < 60) return `${numeric.toFixed(2)} s`;
  const mins = Math.floor(numeric / 60);
  const remain = Math.round(numeric % 60);
  return `${mins} 分 ${remain} 秒`;
}

function sceneDisplayName(scene) {
  const mapping = {
    gaussian_noise: '高斯噪声',
    low_light: '低照度',
    normal: '原始场景',
    partial_crop: '局部裁剪',
  };
  return mapping[scene] || scene || '-';
}

function setElementText(element, value) {
  if (!element) return;
  element.textContent = value;
}

function findModelEntry(models = [], targetName = '') {
  return models.find((item) => item.display_name === targetName || item.model_name === targetName) || null;
}

function renderSummaryMetrics() {
  if (!comparisonCache) return;

  const models = comparisonCache.models || [];
  const bestModel = findModelEntry(models, comparisonCache.best_accuracy_model);
  const fastestModel = findModelEntry(models, comparisonCache.fastest_model);

  setElementText(overviewBestModel, comparisonCache.best_accuracy_model || '-');
  setElementText(reportBestModel, comparisonCache.best_accuracy_model || '-');
  setElementText(
    overviewBestAccuracy,
    bestModel ? `测试准确率 ${bestModel.test_accuracy}%` : '暂无测试准确率'
  );
  setElementText(
    reportBestAccuracy,
    bestModel ? `测试准确率 ${bestModel.test_accuracy}%` : '暂无测试准确率'
  );

  setElementText(overviewFastestModel, comparisonCache.fastest_model || '-');
  setElementText(reportFastestModel, comparisonCache.fastest_model || '-');
  setElementText(
    overviewFastestFps,
    fastestModel ? `${fastestModel.fps} FPS` : '暂无 FPS 数据'
  );
  setElementText(
    reportFastestFps,
    fastestModel ? `${fastestModel.fps} FPS` : '暂无 FPS 数据'
  );

  const datasetText = formatDatasetSize(comparisonCache.dataset_size);
  setElementText(overviewDatasetSize, datasetText);
  setElementText(reportDatasetSize, datasetText);

  if (trainingSummaryCache && trainingSummaryCache.length) {
    const totalSeconds = trainingSummaryCache.reduce((sum, item) => sum + Number(item.training_seconds || 0), 0);
    const text = formatSeconds(totalSeconds);
    setElementText(overviewTrainingTotal, text);
    setElementText(reportTrainingTotal, text);
  }

  if (reportConclusionNote) {
    if (robustnessCache && (robustnessCache.items || []).length) {
      const summary = robustnessCache.items
        .map((item) => `${sceneDisplayName(item.scene)}最佳：${item.best_model}`)
        .join('；');
      reportConclusionNote.textContent = `评估结论：${comparisonCache.best_accuracy_model || '-'} 在综合精度上表现最好，${comparisonCache.fastest_model || '-'} 在推理速度上最优。复杂场景结果为：${summary}。`;
    } else {
      reportConclusionNote.textContent = comparisonCache.note || '评估结论整理中。';
    }
  }
}

function setHealthUI(ok, timeText, labelText) {
  if (!healthPill) return;
  healthPill.classList.remove('status-ok', 'status-bad');
  healthPill.classList.add(ok ? 'status-ok' : 'status-bad');
  healthPill.innerHTML = `
    <span class="health-dot"></span>
    <span class="health-text">${labelText}</span>
    <span class="health-time">${timeText}</span>
  `;
}

async function checkHealth() {
  if (!healthPill) return;
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const ok = data.status === 'ok';
    setHealthUI(ok, `更新于 ${formatTime(data.timestamp)} · v${data.version || '-'}`, ok ? '系统运行正常' : '系统状态异常');
  } catch (err) {
    setHealthUI(false, '健康检查未通过', '系统暂不可达');
  }
}

function renderTip(data) {
  if (!resultTip) return;
  resultTip.classList.remove('hidden');

  if (data.predicted_class === 'unknown') {
    resultTip.className = 'tip-box warning-box';
    resultTip.textContent = '当前图片未能稳定映射到预设的七类道路目标，建议更换更典型的交通场景图片后重试。';
    return;
  }

  const modeText = String(data.model_mode || '');
  const sourceText = modeText.startsWith('hybrid-road7-checkpoint+imagenet:')
    ? '当前结果使用了训练权重与 ImageNet 先验融合纠偏'
    : (modeText.startsWith('trained-road7-checkpoint:')
      ? '当前结果来自训练后的七类专用权重'
      : '当前结果来自 ImageNet 回退映射');
  const camText = data.gradcam_enabled ? '并已生成 Grad-CAM 热力图' : '本次未生成 Grad-CAM 热力图';
  resultTip.className = 'tip-box success-box';
  resultTip.textContent = `识别结果为 ${data.predicted_label}，${camText}。${sourceText}。`;
}

function renderDetections(items = []) {
  if (!detectionsEl) return;
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
    const mapped = item.display_label || item.label || item.class_name || '-';
    el.innerHTML = `<strong>Top-${index + 1}</strong><div>${raw}</div><div>映射结果：${mapped}</div><div>置信度：${score}</div>`;
    detectionsEl.appendChild(el);
  });
}

function renderResult(data, imageSrc) {
  if (!resultPanel || !previewImage || !mainLabel || !mainScore || !modelMode) return;

  resultPanel.classList.remove('hidden');
  previewImage.src = imageSrc || data.image_url || '';
  mainLabel.textContent = data.predicted_label || '-';
  mainScore.textContent = Number(data.confidence || 0).toFixed(3);
  modelMode.textContent = `推理模式：${modeLabel(data.model_mode)} | 当前模型：${data.model_display_name || data.model_name || '-'}`;
  renderTip(data);

  if (annotatedWrapper && annotatedImage) {
    if (data.annotated_url) {
      annotatedWrapper.classList.remove('hidden');
      annotatedImage.src = `${data.annotated_url}?t=${Date.now()}`;
    } else {
      annotatedWrapper.classList.add('hidden');
      annotatedImage.src = '';
    }
  }

  renderDetections(data.detections || []);
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderHistory(items = []) {
  if (!historyList) return;
  historyList.innerHTML = '';
  if (!items.length) {
    historyList.innerHTML = '<div class="history-item">当前暂无识别记录。</div>';
    return;
  }

  items.forEach((item) => {
    const el = document.createElement('article');
    el.className = 'history-item';
    const img = item.annotated_url || item.image_url;
    el.innerHTML = `
      <img class="history-thumb" src="${img}" alt="${item.filename}" />
      <h3>${item.predicted_label}</h3>
      <p>文件：${item.filename}</p>
      <p>置信度：${Number(item.confidence || 0).toFixed(3)}</p>
      <p>模型：${modeLabel(item.model_mode)}</p>
      <p>时间：${formatTime(item.created_at)}</p>
    `;
    historyList.appendChild(el);
  });
}

function renderComparison(data = {}) {
  comparisonCache = data;
  if (!comparisonBody || !comparisonSummary) {
    renderSummaryMetrics();
    return;
  }

  const models = data.models || [];
  const summaryParts = [];
  if (data.dataset_name) summaryParts.push(`数据集：${data.dataset_name}`);
  if (data.dataset_size) summaryParts.push(`样本量：${formatDatasetSize(data.dataset_size)}`);
  if (data.best_accuracy_model) summaryParts.push(`准确率最优：${data.best_accuracy_model}`);
  if (data.fastest_model) summaryParts.push(`速度最优：${data.fastest_model}`);
  comparisonSummary.textContent = summaryParts.join(' ｜ ') || '暂无离线评估总结。';

  comparisonBody.innerHTML = '';
  if (!models.length) {
    comparisonBody.innerHTML = '<tr><td colspan="7">尚未读取到评估结果。</td></tr>';
    return;
  }

  models.forEach((item) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${item.display_name || item.model_name || '-'}</td>
      <td>${item.parameters_million || '-'} M</td>
      <td>${item.val_accuracy || '-'}%</td>
      <td>${item.test_accuracy || '-'}%</td>
      <td>${item.avg_inference_ms || '-'} ms</td>
      <td>${item.fps || '-'}</td>
      <td>${item.note || '-'}</td>
    `;
    comparisonBody.appendChild(tr);
  });

  renderSummaryMetrics();
}

function renderRobustness(data = {}) {
  robustnessCache = data;
  if (!robustnessCards) {
    renderSummaryMetrics();
    return;
  }
  robustnessCards.innerHTML = '';
  const items = data.items || [];
  if (!items.length) return;

  items.forEach((item) => {
    const card = document.createElement('article');
    card.className = 'stat-card';
    card.innerHTML = `<h3>${sceneDisplayName(item.scene)}</h3><p>${item.best_model}</p><div class="muted">${item.summary}</div>`;
    robustnessCards.appendChild(card);
  });

  renderSummaryMetrics();
}

function renderTrainingSummary(items = []) {
  trainingSummaryCache = items;
  if (!trainingSummaryBody || !trainingSummaryNote) {
    renderSummaryMetrics();
    return;
  }

  trainingSummaryBody.innerHTML = '';
  if (!items.length) {
    trainingSummaryBody.innerHTML = '<tr><td colspan="5">尚未读取到训练摘要。</td></tr>';
    trainingSummaryNote.textContent = '训练摘要不可用，请先运行训练脚本。';
    return;
  }

  const totalSeconds = items.reduce((sum, item) => sum + Number(item.training_seconds || 0), 0);
  trainingSummaryNote.textContent = `共读取 ${items.length} 个模型的训练摘要，累计训练时间约 ${formatSeconds(totalSeconds)}。`;

  items.forEach((item) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${modelDisplayName(item.model_name)}</td>
      <td>${item.epochs ?? '-'}</td>
      <td>${item.best_val_acc ?? '-'}%</td>
      <td>${formatSeconds(item.training_seconds)}</td>
      <td>${item.checkpoint || '-'}</td>
    `;
    trainingSummaryBody.appendChild(tr);
  });

  renderSummaryMetrics();
}

function renderStressResults(items = []) {
  if (!stressResults) return;

  stressResults.innerHTML = '';
  if (!items.length) {
    stressResults.innerHTML = '<div class="history-item">尚未执行复杂场景压力测试。</div>';
    return;
  }

  items.forEach((item) => {
    const el = document.createElement('article');
    el.className = 'history-item';
    el.innerHTML = `
      <img class="history-thumb" src="${item.image_url}" alt="${item.scene_label}" />
      <h3>${item.scene_label}</h3>
      <p>${item.scene_description || ''}</p>
      <p>预测类别：${item.predicted_label}</p>
      <p>置信度：${Number(item.confidence || 0).toFixed(3)}</p>
      <p>模型：${item.model_display_name || modelDisplayName(item.model_name)}</p>
    `;
    stressResults.appendChild(el);
  });
}

async function loadHistory() {
  if (!historyList) return;
  try {
    const res = await fetch('/api/history?limit=12');
    const data = await res.json();
    renderHistory(data.items || []);
  } catch (err) {
    historyList.innerHTML = '<div class="history-item">识别记录读取失败，请稍后重试。</div>';
  }
}

async function loadReports() {
  const hasReportConsumers = [
    comparisonBody,
    comparisonSummary,
    robustnessCards,
    trainingSummaryBody,
    trainingSummaryNote,
    overviewBestModel,
    reportBestModel,
    reportConclusionNote,
  ].some(Boolean);
  if (!hasReportConsumers) return;

  try {
    const [comparisonRes, robustnessRes, trainingRes] = await Promise.all([
      fetch('/api/reports/model-comparison'),
      fetch('/api/reports/robustness'),
      fetch('/api/reports/training-summary'),
    ]);
    const comparison = await comparisonRes.json();
    const robustness = await robustnessRes.json();
    const training = await trainingRes.json();
    renderComparison(comparison.data || {});
    renderRobustness(robustness.data || {});
    renderTrainingSummary(training.data || []);
  } catch (err) {
    if (comparisonSummary) {
      comparisonSummary.textContent = '评估报告读取失败，请先运行离线训练与评测脚本生成报告。';
    }
    if (trainingSummaryNote) {
      trainingSummaryNote.textContent = '训练摘要读取失败。';
    }
  }
}

async function submitSingleFile(file, modelName, withGradcam) {
  const formData = new FormData();
  formData.append('file', file, file.name || 'capture.jpg');
  formData.append('model_name', modelName);
  formData.append('with_gradcam', withGradcam ? 'true' : 'false');

  const res = await fetch('/api/classify', { method: 'POST', body: formData });
  const data = await res.json();
  if (!res.ok || !data.success) throw new Error(data.detail || '识别失败');
  return data;
}

async function submitSample(sampleName, modelName, withGradcam) {
  const formData = new FormData();
  formData.append('sample_name', sampleName);
  formData.append('model_name', modelName);
  formData.append('with_gradcam', withGradcam ? 'true' : 'false');

  const res = await fetch('/api/classify/sample', { method: 'POST', body: formData });
  const data = await res.json();
  if (!res.ok || !data.success) throw new Error(data.detail || '样例识别失败');
  return data;
}

async function submitStressTest(file, modelName) {
  const formData = new FormData();
  formData.append('file', file, file.name || 'stress.jpg');
  formData.append('model_name', modelName);

  const res = await fetch('/api/classify/stress-test', { method: 'POST', body: formData });
  const data = await res.json();
  if (!res.ok || !data.success) throw new Error(data.detail || '压力测试失败');
  return data;
}

function initSinglePage() {
  if (!singleUploadForm || !singleFileInput || !singleModelSelect || !singleGradcamCheckbox) return;

  singleUploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!singleFileInput.files || !singleFileInput.files[0]) {
      alert('请先选择待识别图片');
      return;
    }

    const file = singleFileInput.files[0];
    const previewUrl = URL.createObjectURL(file);
    const submitBtn = singleUploadForm.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = '识别处理中…';

    try {
      const data = await submitSingleFile(file, singleModelSelect.value, singleGradcamCheckbox.checked);
      renderResult(data, previewUrl);
      await loadHistory();
    } catch (err) {
      alert(err.message || '识别失败，请稍后重试');
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = '提交图片并开始识别';
    }
  });

  if (refreshHistoryBtn) {
    refreshHistoryBtn.addEventListener('click', loadHistory);
  }

  sampleRunButtons.forEach((button) => {
    button.addEventListener('click', async () => {
      const sampleName = button.dataset.sampleName;
      if (!sampleName) return;

      const originalText = button.textContent;
      button.disabled = true;
      button.textContent = '识别处理中…';

      try {
        const data = await submitSample(sampleName, singleModelSelect.value, singleGradcamCheckbox.checked);
        renderResult(data, data.image_url);
        await loadHistory();
      } catch (err) {
        alert(err.message || '样例识别失败，请稍后重试');
      } finally {
        button.disabled = false;
        button.textContent = originalText;
      }
    });
  });

  loadHistory();
}

function initBatchPage() {
  if (!batchForm || !batchFilesInput || !scenarioModelSelect || !batchResults) return;

  batchForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!batchFilesInput.files || !batchFilesInput.files.length) {
      alert('请先选择批量图片');
      return;
    }

    const formData = new FormData();
    Array.from(batchFilesInput.files).forEach((file) => formData.append('files', file, file.name));
    formData.append('model_name', scenarioModelSelect.value);
    formData.append('with_gradcam', 'false');

    const submitBtn = batchForm.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = '批量处理中…';
    batchResults.innerHTML = '';

    try {
      const res = await fetch('/api/classify/batch', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.detail || '批量识别失败');

      if (!(data.items || []).length) {
        batchResults.innerHTML = '<div class="detection-item">没有可展示的批量结果。</div>';
      } else {
        data.items.forEach((item) => {
          const el = document.createElement('div');
          el.className = 'detection-item';
          if (item.success === false) {
            el.innerHTML = `<strong>${item.filename || '未知文件'}</strong><div>状态：失败</div><div>${item.detail || '-'}</div>`;
          } else {
            el.innerHTML = `<strong>${item.filename}</strong><div>类别：${item.predicted_label}</div><div>置信度：${Number(item.confidence || 0).toFixed(3)}</div><div>模型：${item.model_display_name || item.model_name}</div>`;
          }
          batchResults.appendChild(el);
        });
      }
    } catch (err) {
      batchResults.innerHTML = `<div class="detection-item">${err.message || '批量识别失败，请稍后重试。'}</div>`;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = '开始批量识别';
    }
  });
}

function initStressPage() {
  if (!stressForm || !stressFileInput || !scenarioModelSelect || !stressResults) return;

  stressForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!stressFileInput.files || !stressFileInput.files[0]) {
      alert('请先选择一张图片，再执行复杂场景测试');
      return;
    }

    const file = stressFileInput.files[0];
    const submitBtn = stressForm.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = '压力测试处理中…';
    stressResults.innerHTML = '<div class="history-item">正在生成低照度、噪声和局部裁剪样本…</div>';

    try {
      const data = await submitStressTest(file, scenarioModelSelect.value);
      renderStressResults(data.items || []);
    } catch (err) {
      stressResults.innerHTML = `<div class="history-item">${err.message || '复杂场景压力测试失败。'}</div>`;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = '执行复杂场景压力测试';
    }
  });

  renderStressResults([]);
}

async function startCamera() {
  if (!cameraVideo) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('当前浏览器不支持摄像头调用');
    return;
  }

  try {
    if (currentStream) return;
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false,
    });
    cameraVideo.srcObject = currentStream;
  } catch (err) {
    alert('打开摄像头失败，请检查浏览器授权');
  }
}

async function captureFromCamera() {
  if (!cameraVideo || !cameraCanvas || !scenarioModelSelect) return;

  if (!currentStream) {
    await startCamera();
  }
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
    alert('摄像头画面尚未准备完成，请稍后再试');
    return;
  }

  cameraCanvas.width = cameraVideo.videoWidth;
  cameraCanvas.height = cameraVideo.videoHeight;
  const ctx = cameraCanvas.getContext('2d');
  ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

  const blob = await new Promise((resolve) => cameraCanvas.toBlob(resolve, 'image/jpeg', 0.92));
  if (!blob) {
    alert('抓拍失败，请重试');
    return;
  }

  captureCameraBtn.disabled = true;
  captureCameraBtn.textContent = '实时识别中…';

  try {
    const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
    const data = await submitSingleFile(file, scenarioModelSelect.value, true);
    alert(`实时识别结果：${data.predicted_label}（置信度 ${Number(data.confidence || 0).toFixed(3)}）`);
  } catch (err) {
    alert(err.message || '实时识别失败');
  } finally {
    captureCameraBtn.disabled = false;
    captureCameraBtn.textContent = '抓拍并识别';
  }
}

function initCameraPage() {
  if (!startCameraBtn || !captureCameraBtn) return;
  startCameraBtn.addEventListener('click', startCamera);
  captureCameraBtn.addEventListener('click', captureFromCamera);
}

checkHealth();
loadReports();
initSinglePage();
initBatchPage();
initStressPage();
initCameraPage();
