// グローバル変数
let currentFile = null;
let isProcessing = false;

// DOM要素
const dropArea = document.getElementById('drop-area');
const fileElem = document.getElementById('fileElem');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('preview-image');
const submitButton = document.getElementById('submit-button');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const flaggedStatus = document.getElementById('flagged-status');
const categoryFlags = document.getElementById('category-flags');
const categoryScores = document.getElementById('category-scores');

// イベントリスナー
document.addEventListener('DOMContentLoaded', () => {
    // ドラッグ＆ドロップイベント
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    dropArea.addEventListener('drop', handleDrop, false);
    
    // ファイル選択イベント
    fileElem.addEventListener('change', handleFiles, false);
    
    // 送信ボタンイベント
    submitButton.addEventListener('click', submitImage, false);
});

// ユーティリティ関数
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    dropArea.classList.add('highlight');
}

function unhighlight() {
    dropArea.classList.remove('highlight');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFiles(e) {
    const files = e.target?.files || e;
    if (files.length) {
        currentFile = files[0];
        if (!isImageFile(currentFile)) {
            alert('画像ファイルのみアップロードできます。');
            return;
        }
        displayPreview(currentFile);
    }
}

function isImageFile(file) {
    return file.type.match('image.*');
}

function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        preview.classList.remove('hidden');
    }
    reader.readAsDataURL(file);
}

// レートリミット関数（連続リクエストを防止）
const rateLimitDelay = 2000; // 2秒
let lastRequestTime = 0;

function checkRateLimit() {
    const now = Date.now();
    const timeSinceLastRequest = now - lastRequestTime;
    
    if (timeSinceLastRequest < rateLimitDelay) {
        return false;
    }
    
    lastRequestTime = now;
    return true;
}

// 画像送信関数
async function submitImage() {
    if (!currentFile || isProcessing) return;
    
    // レートリミットチェック
    if (!checkRateLimit()) {
        alert('リクエストが頻繁すぎます。少し待ってから再試行してください。');
        return;
    }
    
    isProcessing = true;
    showLoading();
    
    try {
        // 画像サイズチェックと必要に応じてリサイズ
        const processedImage = await processImageIfNeeded(currentFile);
        
        // 画像をBase64に変換
        const base64Image = await convertToBase64(processedImage);
        
        // Netlify Functionsにリクエスト
        const response = await fetch('/.netlify/functions/moderate-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64Image })
        });
        
        if (!response.ok) {
            throw new Error(`エラー: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('エラー:', error);
        alert(`エラーが発生しました: ${error.message}`);
    } finally {
        hideLoading();
        isProcessing = false;
    }
}

// 画像をBase64に変換する関数
function convertToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    });
}

// 画像処理（サイズ制限）
async function processImageIfNeeded(file) {
    const MAX_SIZE = 4 * 1024 * 1024; // 4MB
    
    if (file.size <= MAX_SIZE) {
        return file;
    }
    
    // 画像をリサイズ
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;
                
                // アスペクト比を維持しながらサイズを縮小
                const aspectRatio = width / height;
                
                // 目標サイズを計算（元のサイズの約70%から開始）
                let scaleFactor = 0.7;
                
                while (true) {
                    const newWidth = Math.floor(width * scaleFactor);
                    const newHeight = Math.floor(height * scaleFactor);
                    
                    canvas.width = newWidth;
                    canvas.height = newHeight;
                    
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, newWidth, newHeight);
                    
                    // 品質を調整して画像をエクスポート
                    const quality = 0.8;
                    const dataUrl = canvas.toDataURL('image/jpeg', quality);
                    
                    // データURLからBlobを作成してサイズをチェック
                    const byteString = atob(dataUrl.split(',')[1]);
                    const ab = new ArrayBuffer(byteString.length);
                    const ia = new Uint8Array(ab);
                    for (let i = 0; i < byteString.length; i++) {
                        ia[i] = byteString.charCodeAt(i);
                    }
                    const blob = new Blob([ab], { type: 'image/jpeg' });
                    
                    if (blob.size <= MAX_SIZE || scaleFactor < 0.1) {
                        // 新しいファイルを作成
                        const resizedFile = new File([blob], file.name, {
                            type: 'image/jpeg',
                            lastModified: new Date().getTime()
                        });
                        resolve(resizedFile);
                        break;
                    }
                    
                    // さらにサイズを縮小
                    scaleFactor *= 0.8;
                }
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}

// UI表示関数
function showLoading() {
    preview.classList.add('hidden');
    loading.classList.remove('hidden');
    results.classList.add('hidden');
}

function hideLoading() {
    loading.classList.add('hidden');
}

// 結果表示関数
function displayResults(data) {
    // 結果セクションを表示
    results.classList.remove('hidden');
    
    // フラグステータスを設定
    const flagged = data.results[0].flagged;
    flaggedStatus.textContent = flagged ? '検出された(Detected)' : '検出されなかった(Not detected)';
    flaggedStatus.style.color = flagged ? 'var(--danger-color)' : 'var(--success-color)';
    
    // カテゴリフラグテーブルをクリア
    categoryFlags.querySelector('tbody').innerHTML = '';
    
    // カテゴリフラグを表示
    const categories = data.results[0].categories;
    for (const [key, value] of Object.entries(categories)) {
        const row = document.createElement('tr');
        
        const categoryCell = document.createElement('td');
        categoryCell.textContent = getCategoryName(key);
        
        const flagCell = document.createElement('td');
        flagCell.textContent = value ? '✓' : '✗';
        flagCell.style.color = value ? 'var(--danger-color)' : 'var(--success-color)';
        
        row.appendChild(categoryCell);
        row.appendChild(flagCell);
        categoryFlags.querySelector('tbody').appendChild(row);
    }
    
    // カテゴリスコアテーブルをクリア
    categoryScores.querySelector('tbody').innerHTML = '';
    
    // カテゴリスコアを表示（スコア順にソート）
    const scores = data.results[0].category_scores;
    const scoreEntries = Object.entries(scores)
        .map(([key, value]) => ({ key, value }))
        .sort((a, b) => b.value - a.value);
    
    for (const { key, value } of scoreEntries) {
        const row = document.createElement('tr');
        
        const categoryCell = document.createElement('td');
        categoryCell.textContent = getCategoryName(key);
        
        const scoreCell = document.createElement('td');
        scoreCell.textContent = value.toFixed(6);
        
        const barCell = document.createElement('td');
        const barContainer = document.createElement('div');
        barContainer.style.width = '100%';
        barContainer.style.backgroundColor = '#f0f0f0';
        barContainer.style.borderRadius = '2px';
        
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.width = `${Math.min(value * 100, 100)}%`;
        // スコアに応じて色を変更
        if (value > 0.7) {
            bar.style.backgroundColor = 'var(--danger-color)';
        } else if (value > 0.3) {
            bar.style.backgroundColor = 'var(--warning-color)';
        } else {
            bar.style.backgroundColor = 'var(--primary-color)';
        }
        
        barContainer.appendChild(bar);
        barCell.appendChild(barContainer);
        
        row.appendChild(categoryCell);
        row.appendChild(scoreCell);
        row.appendChild(barCell);
        categoryScores.querySelector('tbody').appendChild(row);
    }
    
    // 結果までスクロール
    results.scrollIntoView({ behavior: 'smooth' });
}

// カテゴリ名の日本語と英語の併記表示
function getCategoryName(key) {
    const categoryMap = {
        'harassment': '嫌がらせ (Harassment)',
        'harassment/threatening': '嫌がらせ/脅迫 (Harassment/Threatening)',
        'hate': '憎悪 (Hate)',
        'hate/threatening': '憎悪/脅迫 (Hate/Threatening)',
        'self-harm': '自傷 (Self-harm)',
        'self-harm/intent': '自傷/意図 (Self-harm/Intent)',
        'self-harm/instructions': '自傷/指示 (Self-harm/Instructions)',
        'sexual': '性的内容 (Sexual)',
        'sexual/minors': '性的内容/未成年 (Sexual/Minors)',
        'violence': '暴力 (Violence)',
        'violence/graphic': '暴力/グラフィック (Violence/Graphic)',
        'illicit': '違法行為 (Illicit)',
        'illicit/violent': '違法行為/暴力 (Illicit/Violent)'
    };
    
    return categoryMap[key] || key;
}
