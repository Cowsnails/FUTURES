/**
 * Indicator Rendering System
 *
 * Handles rendering of technical indicators on TradingView Lightweight Charts
 */

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
