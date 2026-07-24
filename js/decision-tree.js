/**
 * ================================================================
 * decision-tree.js
 * Implementasi algoritma CART (Classification and Regression Trees)
 * menggunakan Gini Impurity sebagai kriteria splitting.
 * ================================================================
 */

'use strict';

class DecisionTreeClassifier {
    /**
     * @param {object} options
     * @param {number} options.maxDepth       - Kedalaman maksimal pohon (default: 12)
     * @param {number} options.minSamplesSplit - Minimal sampel untuk split (default: 5)
     * @param {number} options.minSamplesLeaf  - Minimal sampel di leaf node (default: 2)
     */
    constructor(options = {}) {
        this.maxDepth       = options.maxDepth       ?? 12;
        this.minSamplesSplit = options.minSamplesSplit ?? 5;
        this.minSamplesLeaf  = options.minSamplesLeaf  ?? 2;
        this.tree            = null;
        this.classes         = [];
        this.featureNames    = [];
    }

    // ─────────────────────────────────────────────────────────────
    // PUBLIC API
    // ─────────────────────────────────────────────────────────────

    /**
     * Melatih model Decision Tree.
     * @param {Array<object>} data   - Array of row objects {feature: value, ..., label: string}
     * @param {string[]}      features - Nama kolom fitur
     * @param {string}        target   - Nama kolom target
     */
    fit(data, features, target) {
        this.featureNames = features;
        this.classes      = [...new Set(data.map(row => row[target]))].sort();

        const X = data.map(row => features.map(f => row[f]));
        const y = data.map(row => row[target]);

        this.tree = this._buildTree(X, y, 0);
        return this;
    }

    /**
     * Memprediksi kelas untuk satu sampel.
     * @param {number[]} sample - Array nilai fitur sesuai urutan this.featureNames
     * @returns {string}        - Kelas yang diprediksi
     */
    predict(sample) {
        if (!this.tree) throw new Error('Model belum dilatih. Panggil fit() terlebih dahulu.');
        return this._traverse(this.tree, sample);
    }

    /**
     * Mendapatkan probabilitas semua kelas untuk satu sampel.
     * @param {number[]} sample
     * @returns {object} - { "No Diabetes": 0.8, "Type 2": 0.2, ... }
     */
    predictProba(sample) {
        if (!this.tree) throw new Error('Model belum dilatih.');
        return this._traverseProba(this.tree, sample);
    }

    // ─────────────────────────────────────────────────────────────
    // TREE BUILDING
    // ─────────────────────────────────────────────────────────────

    _buildTree(X, y, depth) {
        const n        = y.length;
        const counts   = this._classCounts(y);
        const majority = this._majorityClass(counts);

        // Stopping conditions
        if (
            depth >= this.maxDepth ||
            n < this.minSamplesSplit ||
            Object.keys(counts).length === 1
        ) {
            return this._makeLeaf(counts, majority, n);
        }

        // Find best split
        const split = this._bestSplit(X, y);

        if (split === null) {
            return this._makeLeaf(counts, majority, n);
        }

        const { featureIdx, threshold, leftX, leftY, rightX, rightY } = split;

        // Check min samples leaf constraint
        if (leftY.length < this.minSamplesLeaf || rightY.length < this.minSamplesLeaf) {
            return this._makeLeaf(counts, majority, n);
        }

        return {
            type:       'node',
            featureIdx,
            threshold,
            feature:    this.featureNames[featureIdx],
            left:       this._buildTree(leftX, leftY, depth + 1),
            right:      this._buildTree(rightX, rightY, depth + 1),
            n,
            majority,
        };
    }

    _makeLeaf(counts, majority, n) {
        // Convert counts to probabilities
        const proba = {};
        for (const cls of this.classes) {
            proba[cls] = (counts[cls] ?? 0) / n;
        }
        return { type: 'leaf', prediction: majority, proba, n };
    }

    // ─────────────────────────────────────────────────────────────
    // BEST SPLIT (GINI IMPURITY)
    // ─────────────────────────────────────────────────────────────

    _bestSplit(X, y) {
        const nFeatures = X[0].length;
        const n         = y.length;
        let bestGini    = Infinity;
        let bestSplit   = null;

        for (let fi = 0; fi < nFeatures; fi++) {
            // Collect unique thresholds (midpoints between sorted unique values)
            const values    = [...new Set(X.map(row => row[fi]))].sort((a, b) => a - b);
            const thresholds = [];
            for (let i = 0; i < values.length - 1; i++) {
                thresholds.push((values[i] + values[i + 1]) / 2);
            }

            for (const threshold of thresholds) {
                const leftIdx  = [];
                const rightIdx = [];

                for (let i = 0; i < n; i++) {
                    if (X[i][fi] <= threshold) leftIdx.push(i);
                    else rightIdx.push(i);
                }

                if (leftIdx.length === 0 || rightIdx.length === 0) continue;

                const leftY  = leftIdx.map(i => y[i]);
                const rightY = rightIdx.map(i => y[i]);

                const gini = this._weightedGini(leftY, rightY, n);

                if (gini < bestGini) {
                    bestGini  = gini;
                    bestSplit = {
                        featureIdx: fi,
                        threshold,
                        leftX:  leftIdx.map(i => X[i]),
                        leftY,
                        rightX: rightIdx.map(i => X[i]),
                        rightY,
                        gini,
                    };
                }
            }
        }

        return bestSplit;
    }

    _weightedGini(leftY, rightY, total) {
        const leftGini  = this._gini(leftY);
        const rightGini = this._gini(rightY);
        return (leftY.length / total) * leftGini + (rightY.length / total) * rightGini;
    }

    _gini(y) {
        const n      = y.length;
        if (n === 0) return 0;
        const counts = this._classCounts(y);
        let impurity = 1;
        for (const c in counts) {
            const p  = counts[c] / n;
            impurity -= p * p;
        }
        return impurity;
    }

    // ─────────────────────────────────────────────────────────────
    // PREDICTION TRAVERSAL
    // ─────────────────────────────────────────────────────────────

    _traverse(node, sample) {
        if (node.type === 'leaf') return node.prediction;
        if (sample[node.featureIdx] <= node.threshold) {
            return this._traverse(node.left, sample);
        }
        return this._traverse(node.right, sample);
    }

    _traverseProba(node, sample) {
        if (node.type === 'leaf') return node.proba;
        if (sample[node.featureIdx] <= node.threshold) {
            return this._traverseProba(node.left, sample);
        }
        return this._traverseProba(node.right, sample);
    }

    // ─────────────────────────────────────────────────────────────
    // UTILITIES
    // ─────────────────────────────────────────────────────────────

    _classCounts(y) {
        const counts = {};
        for (const label of y) {
            counts[label] = (counts[label] ?? 0) + 1;
        }
        return counts;
    }

    _majorityClass(counts) {
        return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    }

    /**
     * Menghitung akurasi model pada dataset.
     * @param {Array<number[]>} X
     * @param {string[]}        y
     * @returns {number} - Akurasi antara 0 dan 1
     */
    score(X, y) {
        let correct = 0;
        for (let i = 0; i < X.length; i++) {
            if (this._traverse(this.tree, X[i]) === y[i]) correct++;
        }
        return correct / y.length;
    }
}
