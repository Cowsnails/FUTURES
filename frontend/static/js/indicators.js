/**
 * Indicator Rendering System
 *
 * Handles rendering of technical indicators on TradingView Lightweight Charts
 * Updated: Major Supertrend fix - regime-colored lines, background shading, live updates
 */

// ═══════════════════════════════════════════════════════════════════════════
// SUPERTREND MULTI-TIMEFRAME INDICATOR
// ═══════════════════════════════════════════════════════════════════════════

export class SupertrendIndicator {
    constructor(chart, candleSeries) {
        this.chart = chart;
        this.candleSeries = candleSeries;

        // Settings
        this.settings = {
            factor: 3,
            atrLength: 10,
            largerFactor: 3,
            largerAtrLength: 10,
            largerTimeframeMultiplier: 4,
            showAtrRegime: true,
            atrRegimeLength: 14,
        };

        // Series for rendering - just two fixed series (up + down)
        this.upSeries = null;
        this.downSeries = null;

        // Marker arrays
        this.markers = [];

        // State tracking
        this.lastDirection = 0;
        this.lastLargerDirection = 0;
        this.bars = [];

        // Throttle: only full recalc on new bars, not every tick
        this._pendingUpdate = false;
        this._updateThrottleMs = 500;
        this._lastFullRecalcTime = 0;

        // Colors
        this.colors = {
            bullish: '#4CAF50',
            bearish: '#FF5252',
            buySignal: 'rgba(0, 230, 118, 1)',
            sellSignal: 'rgba(255, 82, 82, 1)',
        };
    }

    calculateTR(high, low, prevClose) {
        if (prevClose === null) return high - low;
        return Math.max(
            high - low,
            Math.abs(high - prevClose),
            Math.abs(low - prevClose)
        );
    }

    calculateATR(bars, period) {
        const atr = new Array(bars.length).fill(null);
        let sum = 0;

        for (let i = 0; i < bars.length; i++) {
            const prevClose = i > 0 ? bars[i - 1].close : null;
            const tr = this.calculateTR(bars[i].high, bars[i].low, prevClose);

            if (i < period) {
                sum += tr;
                if (i === period - 1) {
                    atr[i] = sum / period;
                }
            } else {
                atr[i] = (atr[i - 1] * (period - 1) + tr) / period;
            }
        }

        return atr;
    }

    calculateSupertrend(bars, factor, atrPeriod) {
        const atr = this.calculateATR(bars, atrPeriod);
        const supertrend = new Array(bars.length).fill(null);
        const direction = new Array(bars.length).fill(0);
        let upperBand = new Array(bars.length).fill(null);
        let lowerBand = new Array(bars.length).fill(null);

        for (let i = 0; i < bars.length; i++) {
            if (atr[i] === null) continue;

            const hl2 = (bars[i].high + bars[i].low) / 2;
            const basicUpperBand = hl2 + factor * atr[i];
            const basicLowerBand = hl2 - factor * atr[i];

            if (i === 0 || upperBand[i - 1] === null) {
                upperBand[i] = basicUpperBand;
                lowerBand[i] = basicLowerBand;
            } else {
                upperBand[i] = basicUpperBand < upperBand[i - 1] || bars[i - 1].close > upperBand[i - 1]
                    ? basicUpperBand
                    : upperBand[i - 1];

                lowerBand[i] = basicLowerBand > lowerBand[i - 1] || bars[i - 1].close < lowerBand[i - 1]
                    ? basicLowerBand
                    : lowerBand[i - 1];
            }

            if (i === 0) {
                direction[i] = bars[i].close > upperBand[i] ? 1 : -1;
            } else if (direction[i - 1] === 1) {
                direction[i] = bars[i].close < lowerBand[i] ? -1 : 1;
            } else {
                direction[i] = bars[i].close > upperBand[i] ? 1 : -1;
            }

            supertrend[i] = direction[i] === 1 ? lowerBand[i] : upperBand[i];
        }

        return { supertrend, direction, upperBand, lowerBand };
    }

    detectSignals(direction, largerDirection) {
        const buySignals = [];
        const sellSignals = [];

        for (let i = 1; i < direction.length; i++) {
            const largerBullish = largerDirection[i] === 1;
            const currentBullish = direction[i] === 1;
            const prevBullish = direction[i - 1] === 1;
            const largerPrevBullish = largerDirection[i - 1] === 1;

            const crossToBullish = !prevBullish && currentBullish;
            const largerCrossToBullish = !largerPrevBullish && largerBullish;

            if ((largerBullish && crossToBullish) || (currentBullish && largerCrossToBullish)) {
                buySignals.push(i);
            }

            const wasBothBullish = prevBullish && largerPrevBullish;
            const nowOneBearish = !currentBullish || !largerBullish;

            if (wasBothBullish && nowOneBearish) {
                sellSignals.push(i);
            }
        }

        return { buySignals, sellSignals };
    }

    /**
     * Initialize the indicator with data
     */
    initialize(bars) {
        if (!bars || bars.length < 20) {
            console.warn('Not enough bars for Supertrend calculation');
            return;
        }

        this.remove();
        this.bars = [...bars];

        const { upData, downData, markers } = this._computeAllData(bars);

        this.upSeries = this.chart.addLineSeries({
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
            lineWidth: 2,
            color: this.colors.bullish,
        });
        this.downSeries = this.chart.addLineSeries({
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
            lineWidth: 2,
            color: this.colors.bearish,
        });

        this.upSeries.setData(upData);
        this.downSeries.setData(downData);

        // Apply markers
        this.markers = markers;
        if (this.markers.length > 0 && this.candleSeries) {
            try {
                this.markers.sort((a, b) => a.time - b.time);
                this.candleSeries.setMarkers(this.markers);
            } catch (e) {
                console.warn('Error setting markers:', e);
            }
        }

        console.log(`Supertrend initialized: ${bars.length} bars, ${markers.length} signals`);
    }

    /**
     * Compute all rendering data from bars
     */
    _computeAllData(bars) {
        const st1 = this.calculateSupertrend(bars, this.settings.factor, this.settings.atrLength);

        const aggregatedBars = this.aggregateBars(bars, this.settings.largerTimeframeMultiplier);
        const st2 = this.calculateSupertrend(aggregatedBars, this.settings.largerFactor, this.settings.largerAtrLength);
        const expandedSt2 = this.expandToOriginalTimeframe(st2, bars.length, this.settings.largerTimeframeMultiplier);

        // Two series (up/down), both have a point at every timestamp.
        // Active direction: real color. Inactive: transparent color.
        // Both track the SAME supertrend value so no vertical snap at transitions.
        const upData = [];
        const downData = [];
        const TRANSPARENT = 'rgba(0,0,0,0)';

        for (let i = 0; i < bars.length; i++) {
            if (st1.supertrend[i] === null) continue;
            const time = bars[i].time;
            const val = st1.supertrend[i];

            if (st1.direction[i] === 1) {
                upData.push({ time, value: val, color: this.colors.bullish });
                downData.push({ time, value: val, color: TRANSPARENT });
            } else {
                downData.push({ time, value: val, color: this.colors.bearish });
                upData.push({ time, value: val, color: TRANSPARENT });
            }
        }

        // Detect signals (uses larger TF internally for signal logic)
        const signals = this.detectSignals(st1.direction, expandedSt2.direction);
        const markers = [];

        for (const idx of signals.buySignals) {
            if (idx < bars.length) {
                markers.push({
                    time: bars[idx].time,
                    position: 'belowBar',
                    color: this.colors.buySignal,
                    shape: 'arrowUp',
                    size: 1,
                });
            }
        }

        for (const idx of signals.sellSignals) {
            if (idx < bars.length) {
                markers.push({
                    time: bars[idx].time,
                    position: 'aboveBar',
                    color: this.colors.sellSignal,
                    shape: 'arrowDown',
                    size: 1,
                });
            }
        }

        this.lastDirection = st1.direction[st1.direction.length - 1];
        this.lastLargerDirection = expandedSt2.direction[expandedSt2.direction.length - 1];

        return { upData, downData, markers };
    }

    /**
     * Add a new bar and update the indicator incrementally
     */
    addBar(bar, isNewBar) {
        if (!this.bars || this.bars.length === 0) return;

        if (isNewBar) {
            this.bars.push(bar);
            // Full recalc on new bar - direction may have changed
            this._recalcAndUpdate();
        } else {
            // Update last bar in memory but DON'T recalc every tick
            this.bars[this.bars.length - 1] = bar;

            // Throttled update: max once per 500ms for tick updates
            const now = Date.now();
            if (now - this._lastFullRecalcTime > this._updateThrottleMs) {
                this._recalcAndUpdate();
            }
        }
    }

    _recalcAndUpdate() {
        if (!this.upSeries || !this.bars || this.bars.length < 20) return;
        this._lastFullRecalcTime = Date.now();

        const { upData, downData, markers } = this._computeAllData(this.bars);

        try {
            this.upSeries.setData(upData);
            this.downSeries.setData(downData);

            this.markers = markers;
            if (this.markers.length > 0 && this.candleSeries) {
                this.markers.sort((a, b) => a.time - b.time);
                this.candleSeries.setMarkers(this.markers);
            }
        } catch (e) {
            console.warn('Error updating supertrend:', e);
        }
    }

    aggregateBars(bars, multiplier) {
        const aggregated = [];

        for (let i = 0; i < bars.length; i += multiplier) {
            const chunk = bars.slice(i, Math.min(i + multiplier, bars.length));
            if (chunk.length === 0) continue;

            aggregated.push({
                time: chunk[0].time,
                open: chunk[0].open,
                high: Math.max(...chunk.map(b => b.high)),
                low: Math.min(...chunk.map(b => b.low)),
                close: chunk[chunk.length - 1].close,
                volume: chunk.reduce((sum, b) => sum + (b.volume || 0), 0),
            });
        }

        return aggregated;
    }

    expandToOriginalTimeframe(data, originalLength, multiplier) {
        const expanded = {
            supertrend: new Array(originalLength).fill(null),
            direction: new Array(originalLength).fill(0),
        };

        for (let i = 0; i < data.supertrend.length; i++) {
            const startIdx = i * multiplier;
            const endIdx = Math.min(startIdx + multiplier, originalLength);

            for (let j = startIdx; j < endIdx; j++) {
                expanded.supertrend[j] = data.supertrend[i];
                expanded.direction[j] = data.direction[i];
            }
        }

        return expanded;
    }

    /**
     * Update with new bar data (full recalc)
     */
    update(bars) {
        this.bars = bars.slice();
        this._recalcAndUpdate();
    }

    remove() {
        try {
            if (this.upSeries) {
                this.chart.removeSeries(this.upSeries);
                this.upSeries = null;
            }
            if (this.downSeries) {
                this.chart.removeSeries(this.downSeries);
                this.downSeries = null;
            }
            if (this.candleSeries) {
                this.candleSeries.setMarkers([]);
            }
            this.markers = [];
            this.bars = [];
        } catch (e) {
            console.warn('Error removing supertrend series:', e);
        }
    }

    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
    }
}

// ═══════════════════════════════════════════════════════════════════════════


export class IndicatorRenderer {
    constructor(chart) {
        this.chart = chart;
        this.series = {};
        this.separatePanes = {};
    }

    addIndicator(indicatorId, config, data) {
        console.log(`Adding indicator: ${indicatorId}`, config);

        try {
            if (config.pane === 'main') {
                this._addMainPaneIndicator(indicatorId, config, data);
            } else if (config.pane === 'separate') {
                this._addSeparatePaneIndicator(indicatorId, config, data);
            }
        } catch (error) {
            console.error(`Error adding indicator ${indicatorId}:`, error);
        }
    }

    _addMainPaneIndicator(indicatorId, config, data) {
        if (config.type === 'line') {
            const series = this.chart.addLineSeries({
                color: config.color || '#2962FF',
                lineWidth: config.lineWidth || 2,
                priceLineVisible: config.priceLineVisible !== false,
                lastValueVisible: config.lastValueVisible !== false,
                title: config.title || indicatorId,
            });

            series.setData(data.map(d => ({
                time: d.time,
                value: d.value
            })));

            this.series[indicatorId] = series;

        } else if (config.type === 'bands') {
            this._addBollingerBands(indicatorId, config, data);
        }
    }

    _addBollingerBands(indicatorId, config, data) {
        const color = config.color || '#2962FF';
        const lineWidth = config.lineWidth || 1;

        const middleSeries = this.chart.addLineSeries({
            color: color,
            lineWidth: lineWidth,
            title: config.title || 'BB',
        });
        middleSeries.setData(data.map(d => ({
            time: d.time,
            value: d.middle
        })));

        const upperSeries = this.chart.addLineSeries({
            color: color,
            lineWidth: lineWidth,
            lineStyle: 2,
        });
        upperSeries.setData(data.map(d => ({
            time: d.time,
            value: d.upper
        })));

        const lowerSeries = this.chart.addLineSeries({
            color: color,
            lineWidth: lineWidth,
            lineStyle: 2,
        });
        lowerSeries.setData(data.map(d => ({
            time: d.time,
            value: d.lower
        })));

        this.series[indicatorId] = {
            middle: middleSeries,
            upper: upperSeries,
            lower: lowerSeries,
            type: 'bands'
        };
    }

    _addSeparatePaneIndicator(indicatorId, config, data) {
        if (config.type === 'line') {
            const series = this.chart.addLineSeries({
                color: config.color || '#7B1FA2',
                lineWidth: config.lineWidth || 2,
                priceLineVisible: false,
                lastValueVisible: true,
                title: config.title || indicatorId,
            });

            series.setData(data.map(d => ({
                time: d.time,
                value: d.value
            })));

            this.series[indicatorId] = series;

        } else if (config.type === 'macd') {
            this._addMACD(indicatorId, config, data);

        } else if (config.type === 'stochastic') {
            this._addStochastic(indicatorId, config, data);
        }
    }

    _addMACD(indicatorId, config, data) {
        const macdSeries = this.chart.addLineSeries({
            color: config.macdColor || '#2962FF',
            lineWidth: 2,
            title: config.title || 'MACD',
        });
        macdSeries.setData(data.map(d => ({
            time: d.time,
            value: d.macd
        })));

        const signalSeries = this.chart.addLineSeries({
            color: config.signalColor || '#FF6D00',
            lineWidth: 2,
        });
        signalSeries.setData(data.map(d => ({
            time: d.time,
            value: d.signal
        })));

        const histogramSeries = this.chart.addHistogramSeries({
            priceScaleId: '',
        });
        histogramSeries.setData(data.map(d => ({
            time: d.time,
            value: d.histogram,
            color: d.histogram >= 0 ?
                (config.histogramUpColor || 'rgba(38, 166, 154, 0.5)') :
                (config.histogramDownColor || 'rgba(239, 83, 80, 0.5)')
        })));

        this.series[indicatorId] = {
            macd: macdSeries,
            signal: signalSeries,
            histogram: histogramSeries,
            type: 'macd'
        };
    }

    _addStochastic(indicatorId, config, data) {
        const kSeries = this.chart.addLineSeries({
            color: config.kColor || '#2962FF',
            lineWidth: 2,
            title: config.title || 'Stochastic',
        });
        kSeries.setData(data.map(d => ({
            time: d.time,
            value: d.k
        })));

        const dSeries = this.chart.addLineSeries({
            color: config.dColor || '#FF6D00',
            lineWidth: 2,
        });
        dSeries.setData(data.map(d => ({
            time: d.time,
            value: d.d
        })));

        this.series[indicatorId] = {
            k: kSeries,
            d: dSeries,
            type: 'stochastic'
        };
    }

    updateIndicator(indicatorId, newPoint) {
        if (!this.series[indicatorId]) {
            console.warn(`Indicator ${indicatorId} not found`);
            return;
        }

        const series = this.series[indicatorId];

        try {
            if (series.type === 'bands') {
                series.middle.update({ time: newPoint.time, value: newPoint.middle });
                series.upper.update({ time: newPoint.time, value: newPoint.upper });
                series.lower.update({ time: newPoint.time, value: newPoint.lower });

            } else if (series.type === 'macd') {
                series.macd.update({ time: newPoint.time, value: newPoint.macd });
                series.signal.update({ time: newPoint.time, value: newPoint.signal });
                series.histogram.update({
                    time: newPoint.time,
                    value: newPoint.histogram,
                    color: newPoint.histogram >= 0 ?
                        'rgba(38, 166, 154, 0.5)' :
                        'rgba(239, 83, 80, 0.5)'
                });

            } else if (series.type === 'stochastic') {
                series.k.update({ time: newPoint.time, value: newPoint.k });
                series.d.update({ time: newPoint.time, value: newPoint.d });

            } else {
                series.update({ time: newPoint.time, value: newPoint.value });
            }

        } catch (error) {
            console.error(`Error updating indicator ${indicatorId}:`, error);
        }
    }

    removeIndicator(indicatorId) {
        if (!this.series[indicatorId]) {
            return;
        }

        const series = this.series[indicatorId];

        try {
            if (series.type === 'bands') {
                this.chart.removeSeries(series.middle);
                this.chart.removeSeries(series.upper);
                this.chart.removeSeries(series.lower);

            } else if (series.type === 'macd') {
                this.chart.removeSeries(series.macd);
                this.chart.removeSeries(series.signal);
                this.chart.removeSeries(series.histogram);

            } else if (series.type === 'stochastic') {
                this.chart.removeSeries(series.k);
                this.chart.removeSeries(series.d);

            } else {
                this.chart.removeSeries(series);
            }

            delete this.series[indicatorId];
            console.log(`Removed indicator: ${indicatorId}`);

        } catch (error) {
            console.error(`Error removing indicator ${indicatorId}:`, error);
        }
    }

    removeAllIndicators() {
        const indicators = Object.keys(this.series);
        indicators.forEach(id => this.removeIndicator(id));
    }

    getActiveIndicators() {
        return Object.keys(this.series);
    }
}


/**
 * Indicator Manager
 */
export class IndicatorManager {
    constructor(apiBaseUrl = '') {
        this.apiBaseUrl = apiBaseUrl;
        this.activeIndicators = [];
    }

    async getAvailableIndicators() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/indicators`);
            const data = await response.json();
            return data.indicators;
        } catch (error) {
            console.error('Error fetching available indicators:', error);
            return [];
        }
    }

    async getActiveIndicators() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/indicators/active`);
            const data = await response.json();
            this.activeIndicators = data.indicators;
            return this.activeIndicators;
        } catch (error) {
            console.error('Error fetching active indicators:', error);
            return [];
        }
    }

    async addIndicator(type, params = {}) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/indicators/${type}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                throw new Error(`Failed to add indicator: ${response.statusText}`);
            }

            const data = await response.json();
            return data.indicator;

        } catch (error) {
            console.error(`Error adding indicator ${type}:`, error);
            return null;
        }
    }

    async removeIndicator(indicatorId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/indicators/${indicatorId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error(`Failed to remove indicator: ${response.statusText}`);
            }

            return true;

        } catch (error) {
            console.error(`Error removing indicator ${indicatorId}:`, error);
            return false;
        }
    }

    async calculateIndicators(symbol) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/indicators/calculate/${symbol}`);

            if (!response.ok) {
                throw new Error(`Failed to calculate indicators: ${response.statusText}`);
            }

            const data = await response.json();
            return data.indicators;

        } catch (error) {
            console.error(`Error calculating indicators for ${symbol}:`, error);
            return null;
        }
    }
}
