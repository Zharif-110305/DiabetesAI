'use strict';
const GENDER_MAP = {
    'Male': 0,
    'Female': 1,
    'Other': 2,
};

const STAGE_MAP = {
    'No Diabetes': 0,
    'Pre-Diabetes': 1,
    'Gestational': 2,
    'Type 1': 3,
    'Type 2': 4,
};

const STAGE_LABELS = Object.keys(STAGE_MAP);


const FEATURE_COLUMNS = [
    'age',
    'gender',                          
    'is_pregnant',
    'physical_activity_minutes_per_week',
    'diet_score',
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
    'glucose_fasting',
    'glucose_postprandial',
];

const TARGET_COLUMN = 'diabetes_stage';
async function loadAndParseCSV(csvPath, onProgress = () => { }) {
    onProgress(5);

    const response = await fetch(csvPath);
    if (!response.ok) {
        throw new Error(`Gagal memuat dataset: ${response.status} ${response.statusText}`);
    }

    onProgress(20);
    const text = await response.text();
    onProgress(40);

    const rows = parseCSV(text);
    onProgress(60);

    const { X, y } = encodeData(rows);
    onProgress(80);

    return { X, y, featureNames: FEATURE_COLUMNS };
}
function parseCSV(text) {
    const lines = text.trim().split('\n');
    const header = lines[0].split(',').map(h => h.trim());
    const rows = [];

    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        const values = line.split(',');
        if (values.length !== header.length) continue; 

        const row = {};
        for (let j = 0; j < header.length; j++) {
            row[header[j]] = values[j].trim();
        }
        rows.push(row);
    }

    return rows;
}
function encodeData(rows) {
    const X = [];
    const y = [];

    for (const row of rows) {
        
        const label = row[TARGET_COLUMN];
        if (!STAGE_LABELS.includes(label)) continue;

        
        const sample = [
            parseFloat(row['age']) || 0,
            GENDER_MAP[row['gender']] ?? 0,
            parseInt(row['is_pregnant']) || 0,
            parseFloat(row['physical_activity_minutes_per_week']) || 0,
            parseFloat(row['diet_score']) || 0,
            parseInt(row['family_history_diabetes']) || 0,
            parseInt(row['hypertension_history']) || 0,
            parseInt(row['cardiovascular_history']) || 0,
            parseFloat(row['bmi']) || 0,
            parseFloat(row['waist_to_hip_ratio']) || 0,
            parseFloat(row['systolic_bp']) || 0,
            parseFloat(row['diastolic_bp']) || 0,
            parseFloat(row['cholesterol_total']) || 0,
            parseFloat(row['hdl_cholesterol']) || 0,
            parseFloat(row['ldl_cholesterol']) || 0,
            parseFloat(row['triglycerides']) || 0,
            parseFloat(row['glucose_fasting']) || 0,
            parseFloat(row['glucose_postprandial']) || 0,
        ];

        
        if (sample.some(isNaN)) continue;

        X.push(sample);
        y.push(label);
    }

    return { X, y };
}
function encodeUserInput(formData) {
    return [
        parseFloat(formData.age),
        GENDER_MAP[formData.gender] ?? 0,
        parseInt(formData.is_pregnant),
        parseFloat(formData.physical_activity_minutes_per_week),
        parseFloat(formData.diet_score),
        parseInt(formData.family_history_diabetes),
        parseInt(formData.hypertension_history),
        parseInt(formData.cardiovascular_history),
        parseFloat(formData.bmi),
        parseFloat(formData.waist_to_hip_ratio),
        parseFloat(formData.systolic_bp),
        parseFloat(formData.diastolic_bp),
        parseFloat(formData.cholesterol_total),
        parseFloat(formData.hdl_cholesterol),
        parseFloat(formData.ldl_cholesterol),
        parseFloat(formData.triglycerides),
        parseFloat(formData.glucose_fasting),
        parseFloat(formData.glucose_postprandial),
    ];
}