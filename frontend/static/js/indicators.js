/**
 * Indicator Rendering System
 *
 * Handles rendering of technical indicators on TradingView Lightweight Charts
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
            largerTimeframeMultiplier: 4,  // e.g., if on 1min, larger = 4min
            showAtrRegime: true,
            atrRegimeLength: 14,
        };

        // Series for rendering
        this.supertrendUpSeries = null;
        this.supertrendDownSeries = null;
        this.largerSupertrendUpSeries = null;
        this.largerSupertrendDownSeries = null;
        this.atrRegimeSeries = null;

        // Marker arrays
        this.markers = [];

        // State tracking
        this.lastDirection = 0;
        this.lastLargerDirection = 0;

        // Colors
        this.colors = {
            bullish: 'rgba(76, 175, 80, 1)',        // Green
            bullishFill: 'rgba(76, 175, 80, 0.15)', // Light green fill
            bearish: 'rgba(255, 82, 82, 1)',        // Red
            bearishFill: 'rgba(255, 82, 82, 0.15)', // Light red fill
            bullishLarger: 'rgba(76, 175, 80, 0.6)',
            bearishLarger: 'rgba(255, 82, 82, 0.6)',
            buySignal: 'rgba(0, 230, 118, 1)',      // Bright green
            sellSignal: 'rgba(255, 82, 82, 1)',     // Red
            hotRegime: 'rgba(76, 175, 80, 0.07)',   // Green tint
            coldRegime: 'rgba(255, 82, 82, 0.07)', // Red tint
        };
    }

    /**
     * Calculate True Range for a single bar
     */
    calculateTR(high, low, prevClose) {
        if (prevClose === null) return high - low;
        return Math.max(
            high - low,
            Math.abs(high - prevClose),
            Math.abs(low - prevClose)
        );
    }

    /**
     * Calculate ATR (Average True Range)
     */
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
                // Wilder's smoothing
                atr[i] = (atr[i - 1] * (period - 1) + tr) / period;
            }
        }

        return atr;
    }

    /**
     * Calculate Supertrend
     */
    calculateSupertrend(bars, factor, atrPeriod) {
        const atr = this.calculateATR(bars, atrPeriod);
        const supertrend = new Array(bars.length).fill(null);
        const direction = new Array(bars.length).fill(0);  // 1 = up, -1 = down

        let upperBand = new Array(bars.length).fill(null);
        let lowerBand = new Array(bars.length).fill(null);

        for (let i = 0; i < bars.length; i++) {
            if (atr[i] === null) continue;

            const hl2 = (bars[i].high + bars[i].low) / 2;
            const basicUpperBand = hl2 + factor * atr[i];
            const basicLowerBand = hl2 - factor * atr[i];

            // Calculate final bands
            if (i === 0 || upperBand[i - 1] === null) {
                upperBand[i] = basicUpperBand;
                lowerBand[i] = basicLowerBand;
            } else {
                // Upper band
                upperBand[i] = basicUpperBand < upperBand[i - 1] || bars[i - 1].close > upperBand[i - 1]
                    ? basicUpperBand
                    : upperBand[i - 1];

                // Lower band
                lowerBand[i] = basicLowerBand > lowerBand[i - 1] || bars[i - 1].close < lowerBand[i - 1]
                    ? basicLowerBand
                    : lowerBand[i - 1];
            }

            // Determine direction
            if (i === 0) {
                direction[i] = bars[i].close > upperBand[i] ? 1 : -1;
            } else if (direction[i - 1] === 1) {
                direction[i] = bars[i].close < lowerBand[i] ? -1 : 1;
            } else {
                direction[i] = bars[i].close > upperBand[i] ? 1 : -1;
            }

            // Set supertrend value based on direction
            supertrend[i] = direction[i] === 1 ? lowerBand[i] : upperBand[i];
        }

        return { supertrend, direction, upperBand, lowerBand };
    }

    /**
     * Calculate ATR Regime (above/below median)
     */
    calculateAtrRegime(bars, period, lookback = 100) {
        const atr = this.calculateATR(bars, period);
        const regime = new Array(bars.length).fill(null);

        for (let i = lookback - 1; i < bars.length; i++) {
            // Get last 'lookback' ATR values
            const atrValues = [];
            for (let j = i - lookback + 1; j <= i; j++) {
                if (atr[j] !== null) atrValues.push(atr[j]);
            }

            if (atrValues.length >= 20) {
                // Calculate median
                const sorted = [...atrValues].sort((a, b) => a - b);
                const mid = Math.floor(sorted.length / 2);
                const median = sorted.length % 2 === 0
                    ? (sorted[mid - 1] + sorted[mid]) / 2
                    : sorted[mid];

                // Check if current ATR is above median
                regime[i] = atr[i] > median;
            }
        }

        return regime;
    }

    /**
     * Detect buy/sell signals
     */
    detectSignals(direction, largerDirection) {
        const buySignals = [];
        const sellSignals = [];

        for (let i = 1; i < direction.length; i++) {
            const largerBullish = largerDirection[i] === 1;
            const currentBullish = direction[i] === 1;
            const prevBullish = direction[i - 1] === 1;
            const largerPrevBullish = largerDirection[i - 1] === 1;

            // Buy signal: larger TF is bullish AND current TF crosses to bullish
            // OR current TF is bullish AND larger TF crosses to bullish
            const crossToBullish = !prevBullish && currentBullish;
            const largerCrossToBullish = !largerPrevBullish && largerBullish;

            if ((largerBullish && crossToBullish) || (currentBullish && largerCrossToBullish)) {
                buySignals.push(i);
            }

            // Sell signal: was both bullish, now one is bearish
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

        // Remove existing series if any
        this.remove();

        // Calculate current timeframe supertrend
        const st1 = this.calculateSupertrend(bars, this.settings.factor, this.settings.atrLength);

        // For larger timeframe, we aggregate bars and calculate
        const aggregatedBars = this.aggregateBars(bars, this.settings.largerTimeframeMultiplier);
        const st2 = this.calculateSupertrend(aggregatedBars, this.settings.largerFactor, this.settings.largerAtrLength);

        // Expand larger timeframe data back to original timeframe
        const expandedSt2 = this.expandToOriginalTimeframe(st2, bars.length, this.settings.largerTimeframeMultiplier);

        // Calculate ATR regime
        const atrRegime = this.calculateAtrRegime(bars, this.settings.atrRegimeLength);

        // Create supertrend line series (current timeframe)
        this.supertrendUpSeries = this.chart.addLineSeries({
            color: this.colors.bullish,
            lineWidth: 2,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
        });

        this.supertrendDownSeries = this.chart.addLineSeries({
            color: this.colors.bearish,
            lineWidth: 2,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
        });

        // Create larger timeframe supertrend series
        this.largerSupertrendUpSeries = this.chart.addLineSeries({
            color: this.colors.bullishLarger,
            lineWidth: 1,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
            lineStyle: 0,
        });

        this.largerSupertrendDownSeries = this.chart.addLineSeries({
            color: this.colors.bearishLarger,
            lineWidth: 1,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
            lineStyle: 0,
        });

        // Prepare data for series
        const upData = [];
        const downData = [];
        const largerUpData = [];
        const largerDownData = [];

        for (let i = 0; i < bars.length; i++) {
            const time = bars[i].time;

            // Current timeframe supertrend
            if (st1.supertrend[i] !== null) {
                if (st1.direction[i] === 1) {
                    upData.push({ time, value: st1.supertrend[i] });
                } else {
                    downData.push({ time, value: st1.supertrend[i] });
                }
            }

            // Larger timeframe supertrend
            if (expandedSt2.supertrend[i] !== null) {
                if (expandedSt2.direction[i] === 1) {
                    largerUpData.push({ time, value: expandedSt2.supertrend[i] });
                } else {
                    largerDownData.push({ time, value: expandedSt2.supertrend[i] });
                }
            }
        }

        // Set data
        this.supertrendUpSeries.setData(upData);
        this.supertrendDownSeries.setData(downData);
        this.largerSupertrendUpSeries.setData(largerUpData);
        this.largerSupertrendDownSeries.setData(largerDownData);

        // Detect and render signals
        const signals = this.detectSignals(st1.direction, expandedSt2.direction);
        this.markers = [];

        for (const idx of signals.buySignals) {
            if (idx < bars.length) {
                this.markers.push({
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
                this.markers.push({
                    time: bars[idx].time,
                    position: 'aboveBar',
                    color: this.colors.sellSignal,
                    shape: 'arrowDown',
                    size: 1,
                });
            }
        }

        // Apply markers to candle series
        if (this.markers.length > 0 && this.candleSeries) {
            this.candleSeries.setMarkers(this.markers);
        }

        // Store state for updates
        this.lastDirection = st1.direction[st1.direction.length - 1];
        this.lastLargerDirection = expandedSt2.direction[expandedSt2.direction.length - 1];

        console.log(`Supertrend initialized: ${bars.length} bars, ${signals.buySignals.length} buy signals, ${signals.sellSignals.length} sell signals`);
    }

    /**
     * Aggregate bars to larger timeframe
     */
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

    /**
     * Expand aggregated data back to original timeframe
     */
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
     * Update with new bar data
     */
    update(bars) {
        // For simplicity, just reinitialize when data updates
        // A more efficient implementation would update incrementally
        this.initialize(bars);
    }

    /**
     * Remove all indicator series from chart
     */
    remove() {
        try {
            if (this.supertrendUpSeries) {
                this.chart.removeSeries(this.supertrendUpSeries);
                this.supertrendUpSeries = null;
            }
            if (this.supertrendDownSeries) {
                this.chart.removeSeries(this.supertrendDownSeries);
                this.supertrendDownSeries = null;
            }
            if (this.largerSupertrendUpSeries) {
                this.chart.removeSeries(this.largerSupertrendUpSeries);
                this.largerSupertrendUpSeries = null;
            }
            if (this.largerSupertrendDownSeries) {
                this.chart.removeSeries(this.largerSupertrendDownSeries);
                this.largerSupertrendDownSeries = null;
            }
            if (this.candleSeries) {
                this.candleSeries.setMarkers([]);
            }
            this.markers = [];
        } catch (e) {
            console.warn('Error removing supertrend series:', e);
        }
    }

    /**
     * Update settings
     */
    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
    }
}

// ═══════════════════════════════════════════════════════════════════════════


export class IndicatorRenderer {
    constructor(chart) {
        this.chart = chart;
        this.series = {};  // indicator_id -> series object
        this.separatePanes = {};  // pane_id -> pane info
    }

    /**
     * Add an indicator to the chart
     *
     * @param {string} indicatorId - Unique indicator ID
     * @param {object} config - Plot configuration from backend
     * @param {array} data - Indicator data points
     */
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

    /**
     * Add indicator that overlays on main price chart
     */
    _addMainPaneIndicator(indicatorId, config, data) {
        if (config.type === 'line') {
            // Standard line overlay (SMA, EMA, etc.)
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
            // Bollinger Bands (3 lines)
            this._addBollingerBands(indicatorId, config, data);
        }
    }

    /**
     * Add Bollinger Bands (special case with 3 lines)
     */
    _addBollingerBands(indicatorId, config, data) {
        const color = config.color || '#2962FF';
        const lineWidth = config.lineWidth || 1;

        // Middle line
        const middleSeries = this.chart.addLineSeries({
            color: color,
            lineWidth: lineWidth,
            title: config.title || 'BB',
        });
        middleSeries.setData(data.map(d => ({
            time: d.time,
            value: d.middle
        })));

        // Upper line
        const upperSeries = this.chart.addLineSeries({
            color: color,
            lineWidth: lineWidth,
            lineStyle: 2,  // Dashed
        });
        upperSeries.setData(data.map(d => ({
            time: d.time,
            value: d.upper
        })));

        // Lower line
        const lowerSeries = this.chart.addLineSeries({
            color: color,
            lineWidth: lineWidth,
            lineStyle: 2,  // Dashed
        });
        lowerSeries.setData(data.map(d => ({
            time: d.time,
            value: d.lower
        })));

        // Store all three series
        this.series[indicatorId] = {
            middle: middleSeries,
            upper: upperSeries,
            lower: lowerSeries,
            type: 'bands'
        };
    }

    /**
     * Add indicator in separate pane below main chart
     */
    _addSeparatePaneIndicator(indicatorId, config, data) {
        if (config.type === 'line') {
            // Standard oscillator (RSI, CCI, ROC)
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

    /**
     * Add MACD indicator (2 lines + histogram)
     */
    _addMACD(indicatorId, config, data) {
        // MACD line
        const macdSeries = this.chart.addLineSeries({
            color: config.macdColor || '#2962FF',
            lineWidth: 2,
            title: config.title || 'MACD',
        });
        macdSeries.setData(data.map(d => ({
            time: d.time,
            value: d.macd
        })));

        // Signal line
        const signalSeries = this.chart.addLineSeries({
            color: config.signalColor || '#FF6D00',
            lineWidth: 2,
        });
        signalSeries.setData(data.map(d => ({
            time: d.time,
            value: d.signal
        })));

        // Histogram
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

    /**
     * Add Stochastic indicator (%K and %D lines)
     */
    _addStochastic(indicatorId, config, data) {
        // %K line
        const kSeries = this.chart.addLineSeries({
            color: config.kColor || '#2962FF',
            lineWidth: 2,
            title: config.title || 'Stochastic',
        });
        kSeries.setData(data.map(d => ({
            time: d.time,
            value: d.k
        })));

        // %D line
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

    /**
     * Update indicator with new data point
     *
     * @param {string} indicatorId - Indicator ID
     * @param {object} newPoint - New data point
     */
    updateIndicator(indicatorId, newPoint) {
        if (!this.series[indicatorId]) {
            console.warn(`Indicator ${indicatorId} not found`);
            return;
        }

        const series = this.series[indicatorId];

        try {
            if (series.type === 'bands') {
                // Update Bollinger Bands
                series.middle.update({ time: newPoint.time, value: newPoint.middle });
                series.upper.update({ time: newPoint.time, value: newPoint.upper });
                series.lower.update({ time: newPoint.time, value: newPoint.lower });

            } else if (series.type === 'macd') {
                // Update MACD
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
                // Update Stochastic
                series.k.update({ time: newPoint.time, value: newPoint.k });
                series.d.update({ time: newPoint.time, value: newPoint.d });

            } else {
                // Standard line series
                series.update({ time: newPoint.time, value: newPoint.value });
            }

        } catch (error) {
            console.error(`Error updating indicator ${indicatorId}:`, error);
        }
    }

    /**
     * Remove an indicator from the chart
     *
     * @param {string} indicatorId - Indicator ID
     */
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

    /**
     * Remove all indicators
     */
    removeAllIndicators() {
        const indicators = Object.keys(this.series);
        indicators.forEach(id => this.removeIndicator(id));
    }

    /**
     * Get list of active indicator IDs
     */
    getActiveIndicators() {
        return Object.keys(this.series);
    }
}


/**
 * Indicator Manager
 *
 * Manages indicator state and API communication
 */
export class IndicatorManager {
    constructor(apiBaseUrl = '') {
        this.apiBaseUrl = apiBaseUrl;
        this.activeIndicators = [];
    }

    /**
     * Get list of available indicators from backend
     */
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

    /**
     * Get list of active indicators
     */
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

    /**
     * Add an indicator
     */
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

    /**
     * Remove an indicator
     */
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

    /**
     * Calculate indicators for a symbol
     */
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
