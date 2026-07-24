/**
 * ================================================================
 * app.js
 * Logika utama aplikasi: inisialisasi, training, prediksi, dan UI.
 * ================================================================
 */

'use strict';

// ─────────────────────────────────────────────────────────────
// KONFIGURASI HASIL — Deskripsi & Saran per Kategori
// ─────────────────────────────────────────────────────────────

const RESULT_CONFIG = {
    'No Diabetes': {
        icon    : '✅',
        color   : '#10b981',
        gradient: 'linear-gradient(135deg, #064e3b, #065f46)',
        badge   : 'Tidak Diabetes',
        summary : 'Kadar gula darah Anda berada dalam batas normal.',
        tips    : [
            'Pertahankan pola makan sehat dan seimbang',
            'Lanjutkan rutinitas olahraga yang teratur',
            'Lakukan pemeriksaan kesehatan tahunan',
            'Jaga berat badan ideal',
        ],
    },
    'Pre-Diabetes': {
        icon    : '⚠️',
        color   : '#f59e0b',
        gradient: 'linear-gradient(135deg, #451a03, #78350f)',
        badge   : 'Pre-Diabetes',
        summary : 'Kadar gula darah Anda sedikit di atas normal. Perlu perhatian khusus.',
        tips    : [
            'Kurangi konsumsi karbohidrat olahan dan gula',
            'Tingkatkan aktivitas fisik minimal 150 menit/minggu',
            'Turunkan berat badan jika kelebihan berat',
            'Konsultasikan dengan dokter untuk pemantauan rutin',
        ],
    },
    'Gestational': {
        icon    : '🤰',
        color   : '#38bdf8',
        gradient: 'linear-gradient(135deg, #0c4a6e, #075985)',
        badge   : 'Diabetes Gestasional',
        summary : 'Kondisi diabetes yang terjadi selama masa kehamilan.',
        tips    : [
            'Pantau kadar gula darah secara rutin bersama dokter kandungan',
            'Ikuti diet khusus yang direkomendasikan dokter',
            'Lakukan olahraga ringan yang aman untuk ibu hamil',
            'Kondisi ini biasanya membaik setelah melahirkan',
        ],
    },
    'Type 1': {
        icon    : '💉',
        color   : '#f97316',
        gradient: 'linear-gradient(135deg, #431407, #7c2d12)',
        badge   : 'Diabetes Tipe 1',
        summary : 'Kondisi autoimun di mana pankreas tidak memproduksi insulin.',
        tips    : [
            'Terapi insulin adalah penanganan utama',
            'Pantau gula darah secara ketat setiap hari',
            'Pelajari cara mengelola hipoglikemia dan hiperglikemia',
            'Konsultasikan dengan endokrinologis secara rutin',
        ],
    },
    'Type 2': {
        icon    : '🩺',
        color   : '#ef4444',
        gradient: 'linear-gradient(135deg, #450a0a, #7f1d1d)',
        badge   : 'Diabetes Tipe 2',
        summary : 'Kondisi resistensi insulin yang paling umum terjadi.',
        tips    : [
            'Ubah pola makan: kurangi gula, perbanyak serat',
            'Olahraga teratur dapat meningkatkan sensitivitas insulin',
            'Ikuti pengobatan yang diresepkan dokter',
            'Pantau komplikasi: tekanan darah, kolesterol, ginjal',
        ],
    },
};

// ─────────────────────────────────────────────────────────────
// STATE GLOBAL
// ─────────────────────────────────────────────────────────────

let model        = null;
let modelReady   = false;
let trainingData = null;

// ─────────────────────────────────────────────────────────────
// INISIALISASI
// ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    initGenderConditional();
    await trainModel();
});

// ─────────────────────────────────────────────────────────────
// TRAINING MODEL
// ─────────────────────────────────────────────────────────────

async function trainModel() {
    const progressBar   = document.getElementById('progress-bar');
    const progressText  = document.getElementById('progress-text');
    const loadingScreen = document.getElementById('loading-screen');
    const appScreen     = document.getElementById('app-screen');
    const statusDetail  = document.getElementById('status-detail');

    function setProgress(val, text, detail = '') {
        progressBar.style.width  = val + '%';
        progressText.textContent = text;
        if (detail) statusDetail.textContent = detail;
    }

    try {
        setProgress(5, 'Memuat dataset...', 'Mengambil data dari server');

        const { X, y, featureNames } = await loadAndParseCSV(
            'DIABETES.csv',
            (pct) => {
                if (pct <= 40) setProgress(pct, 'Memproses dataset...', `${pct}% selesai`);
            }
        );

        trainingData = { X, y, featureNames };

        setProgress(55, 'Mempersiapkan model AI...', `${X.length.toLocaleString()} data berhasil dimuat`);

        // Training di dalam microtask agar UI bisa diperbarui
        await new Promise(resolve => setTimeout(resolve, 80));

        model = new DecisionTreeClassifier({
            maxDepth       : 12,
            minSamplesSplit: 5,
            minSamplesLeaf : 2,
        });

        // Buat object dataset untuk fit()
        const dataset = X.map((row, i) => {
            const obj = {};
            featureNames.forEach((f, fi) => { obj[f] = row[fi]; });
            obj['diabetes_stage'] = y[i];
            return obj;
        });

        setProgress(70, 'Melatih model AI...', 'Mengoptimalkan algoritma untuk akurasi terbaik');
        await new Promise(resolve => setTimeout(resolve, 80));

        model.fit(dataset, featureNames, 'diabetes_stage');

        setProgress(90, 'Mengevaluasi performa...', 'Mengukur akurasi model');
        await new Promise(resolve => setTimeout(resolve, 50));

        const accuracy = model.score(X, y);

        setProgress(100, 'Selesai!', `Akurasi model: ${(accuracy * 100).toFixed(1)}%`);
        await new Promise(resolve => setTimeout(resolve, 600));

        // Update accuracy badge di halaman
        const accBadge = document.getElementById('accuracy-badge');
        if (accBadge) {
            accBadge.textContent = `Akurasi Model: ${(accuracy * 100).toFixed(1)}%`;
        }

        modelReady = true;

        // Transisi ke halaman aplikasi
        loadingScreen.classList.add('hide');
        await new Promise(resolve => setTimeout(resolve, 400));
        loadingScreen.style.display = 'none';
        appScreen.classList.remove('hidden');
        appScreen.classList.add('visible');

    } catch (err) {
        setProgress(0, 'Gagal memuat!', err.message);
        console.error('Training error:', err);
        document.getElementById('loading-error').textContent =
            `Error: ${err.message}. Pastikan web diakses melalui server (bukan file://).`;
        document.getElementById('loading-error').classList.remove('hidden');
    }
}

// ─────────────────────────────────────────────────────────────
// LOGIKA FORM
// ─────────────────────────────────────────────────────────────

/** Tampilkan/sembunyikan field is_pregnant berdasarkan gender */
function initGenderConditional() {
    const genderSelect = document.getElementById('gender');
    const pregnantRow  = document.getElementById('pregnant-row');

    function updatePregnant() {
        const isFemale = genderSelect.value === 'Female';
        pregnantRow.style.display = isFemale ? '' : 'none';
        if (!isFemale) {
            document.getElementById('is_pregnant').value = '0';
        }
    }

    genderSelect.addEventListener('change', updatePregnant);
    updatePregnant();
}

/** Kumpulkan semua nilai form */
function collectFormData() {
    const simpleIds = [
        'age', 'gender',
        'physical_activity_minutes_per_week', 'diet_score',
        'bmi', 'waist_to_hip_ratio',
        'systolic_bp', 'diastolic_bp',
        'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
        'glucose_fasting', 'glucose_postprandial',
    ];

    const data = {};
    for (const id of simpleIds) {
        const el = document.getElementById(id);
        data[id] = el ? el.value.trim() : '';
    }

    // Radio button fields — ambil nilai yang dipilih
    const radioFields = [
        'is_pregnant',
        'family_history_diabetes',
        'hypertension_history',
        'cardiovascular_history',
    ];
    for (const name of radioFields) {
        const checked = document.querySelector(`input[name="${name}"]:checked`);
        // is_pregnant: jika gender bukan Female, paksa 0
        if (name === 'is_pregnant' && data['gender'] !== 'Female') {
            data[name] = '0';
        } else {
            data[name] = checked ? checked.value : '0';
        }
    }

    return data;
}

/** Validasi form — return array pesan error */
function validateForm(data) {
    const errors = [];

    const numFields = {
        age                                : { label: 'Usia', min: 0, max: 120 },
        physical_activity_minutes_per_week : { label: 'Waktu Olahraga', min: 0, max: 1440 },
        diet_score                         : { label: 'Skor Pola Makan', min: 1, max: 10 },
        bmi                                : { label: 'BMI', min: 10, max: 70 },
        waist_to_hip_ratio                 : { label: 'Rasio Pinggang-Pinggul', min: 0.5, max: 2.0 },
        systolic_bp                        : { label: 'Tekanan Darah Sistolik', min: 70, max: 250 },
        diastolic_bp                       : { label: 'Tekanan Darah Diastolik', min: 40, max: 160 },
        cholesterol_total                  : { label: 'Total Kolesterol', min: 50, max: 500 },
        hdl_cholesterol                    : { label: 'Kolesterol HDL', min: 10, max: 200 },
        ldl_cholesterol                    : { label: 'Kolesterol LDL', min: 10, max: 400 },
        triglycerides                      : { label: 'Trigliserida', min: 10, max: 1000 },
        glucose_fasting                    : { label: 'Gula Darah Puasa', min: 50, max: 500 },
        glucose_postprandial               : { label: 'Gula Darah 2 Jam PP', min: 50, max: 600 },
    };

    for (const [key, cfg] of Object.entries(numFields)) {
        const val = parseFloat(data[key]);
        if (isNaN(val)) {
            errors.push(`${cfg.label} harus diisi dengan angka.`);
        } else if (val < cfg.min || val > cfg.max) {
            errors.push(`${cfg.label} harus antara ${cfg.min} dan ${cfg.max}.`);
        }
    }

    if (!data.gender) errors.push('Jenis Kelamin harus dipilih.');

    return errors;
}

// ─────────────────────────────────────────────────────────────
// PREDIKSI
// ─────────────────────────────────────────────────────────────

async function runPrediction() {
    if (!modelReady) {
        showToast('Model masih dalam proses pelatihan, harap tunggu.', 'warning');
        return;
    }

    const formData = collectFormData();
    const errors   = validateForm(formData);

    if (errors.length > 0) {
        showValidationErrors(errors);
        return;
    }

    clearValidationErrors();

    const btn     = document.getElementById('predict-btn');
    const spinner = document.getElementById('btn-spinner');
    btn.disabled  = true;
    spinner.classList.remove('hidden');

    await new Promise(r => setTimeout(r, 300)); // Visual feedback

    try {
        const sample  = encodeUserInput(formData);
        const result  = model.predict(sample);
        const proba   = model.predictProba(sample);

        displayResult(result, proba, formData);
    } catch (err) {
        showToast(`Prediksi gagal: ${err.message}`, 'error');
        console.error(err);
    } finally {
        btn.disabled = false;
        spinner.classList.add('hidden');
    }
}

// ─────────────────────────────────────────────────────────────
// TAMPILKAN HASIL
// ─────────────────────────────────────────────────────────────

function displayResult(prediction, proba, formData) {
    const cfg        = RESULT_CONFIG[prediction];
    const resultPanel = document.getElementById('result-panel');

    // Header
    document.getElementById('result-icon').textContent        = cfg.icon;
    document.getElementById('result-badge').textContent       = cfg.badge;
    document.getElementById('result-badge').style.background  = cfg.color;
    document.getElementById('result-summary').textContent     = cfg.summary;
    resultPanel.style.background                              = cfg.gradient;

    // Tips
    const tipsList = document.getElementById('result-tips');
    tipsList.innerHTML = cfg.tips
        .map(tip => `<li><span class="tip-bullet">→</span>${tip}</li>`)
        .join('');

    // Probability bars
    const probaContainer = document.getElementById('proba-bars');
    probaContainer.innerHTML = '';

    const sortedProba = Object.entries(proba).sort((a, b) => b[1] - a[1]);
    for (const [cls, prob] of sortedProba) {
        const pct     = (prob * 100).toFixed(1);
        const clsCfg  = RESULT_CONFIG[cls];
        const isMain  = cls === prediction;

        probaContainer.innerHTML += `
        <div class="proba-row ${isMain ? 'proba-row--active' : ''}">
          <div class="proba-label">
            <span class="proba-icon">${clsCfg.icon}</span>
            <span>${clsCfg.badge}</span>
            ${isMain ? '<span class="proba-tag">Prediksi</span>' : ''}
          </div>
          <div class="proba-bar-track">
            <div class="proba-bar-fill"
                 style="width: ${pct}%; background: ${clsCfg.color};"
                 data-pct="${pct}">
            </div>
          </div>
          <span class="proba-pct">${pct}%</span>
        </div>`;
    }

    // Disclaimer
    document.getElementById('result-disclaimer').classList.remove('hidden');

    // Show panel
    resultPanel.classList.remove('hidden');
    resultPanel.classList.add('result-appear');

    // Animate bars after paint
    requestAnimationFrame(() => {
        document.querySelectorAll('.proba-bar-fill').forEach(bar => {
            bar.style.transition = 'width 0.8s ease-out';
        });
    });

    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─────────────────────────────────────────────────────────────
// VALIDASI UI
// ─────────────────────────────────────────────────────────────

function showValidationErrors(errors) {
    const container = document.getElementById('validation-errors');
    const box       = document.getElementById('validation-box');
    container.innerHTML = errors.map(e => `<li>${e}</li>`).join('');
    box.classList.remove('hidden');
    box.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function clearValidationErrors() {
    const container = document.getElementById('validation-errors');
    const box       = document.getElementById('validation-box');
    if (container) container.innerHTML = '';
    if (box)       box.classList.add('hidden');
}

// ─────────────────────────────────────────────────────────────
// RESET FORM
// ─────────────────────────────────────────────────────────────

function resetForm() {
    document.getElementById('diabetes-form').reset();
    document.getElementById('result-panel').classList.add('hidden');
    document.getElementById('result-panel').classList.remove('result-appear');
    clearValidationErrors();
    initGenderConditional();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ─────────────────────────────────────────────────────────────
// TOAST NOTIFICATIONS
// ─────────────────────────────────────────────────────────────

function showToast(message, type = 'info') {
    const toast    = document.getElementById('toast');
    toast.textContent   = message;
    toast.className     = `toast toast--${type} toast--show`;
    setTimeout(() => toast.classList.remove('toast--show'), 3500);
}
