/**
 * Scalping Decision Engine
 *
 * Full indicator stack + regime detection + confluence scoring + decision tree
 * for intraday scalping on MNQ/NQ, MGC/GC futures.
 */

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 1: CORE INDICATOR CALCULATIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * EMA - Exponential Moving Average
 * @param {number[]} values - Close prices
 * @param {number} period
 * @returns {number[]} - EMA values (null until enough data)
 */
export function calcEMA(values, period) {
    const ema = new Array(values.length).fill(null);
    const k = 2 / (period + 1);
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        if (i < period - 1) {
            sum += values[i];
            continue;
        }
        if (i === period - 1) {
            sum += values[i];
            ema[i] = sum / period;
            continue;
        }
        ema[i] = values[i] * k + ema[i - 1] * (1 - k);
    }
    return ema;
}

/**
 * SMA - Simple Moving Average
 * @param {number[]} values
 * @param {number} period
 * @returns {number[]}
 */
export function calcSMA(values, period) {
    const sma = new Array(values.length).fill(null);
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        sum += values[i];
        if (i >= period) sum -= values[i - period];
        if (i >= period - 1) sma[i] = sum / period;
    }
    return sma;
}

/**
 * ATR - Average True Range (Wilder's smoothing)
 * @param {Object[]} bars - OHLCV bars
 * @param {number} period
 * @returns {number[]}
 */
export function calcATR(bars, period) {
    const atr = new Array(bars.length).fill(null);
    let sum = 0;
    for (let i = 0; i < bars.length; i++) {
        const prevClose = i > 0 ? bars[i - 1].close : bars[i].close;
        const tr = Math.max(
            bars[i].high - bars[i].low,
            Math.abs(bars[i].high - prevClose),
            Math.abs(bars[i].low - prevClose)
        );
        if (i < period) {
            sum += tr;
            if (i === period - 1) atr[i] = sum / period;
        } else {
            atr[i] = (atr[i - 1] * (period - 1) + tr) / period;
        }
    }
    return atr;
}

/**
 * RSI - Relative Strength Index (Wilder's smoothing)
 * @param {number[]} closes
 * @param {number} period
 * @returns {number[]}
 */
export function calcRSI(closes, period) {
    const rsi = new Array(closes.length).fill(null);
    let avgGain = 0, avgLoss = 0;

    for (let i = 1; i < closes.length; i++) {
        const change = closes[i] - closes[i - 1];
        const gain = change > 0 ? change : 0;
        const loss = change < 0 ? -change : 0;

        if (i <= period) {
            avgGain += gain;
            avgLoss += loss;
            if (i === period) {
                avgGain /= period;
                avgLoss /= period;
                rsi[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
            }
        } else {
            avgGain = (avgGain * (period - 1) + gain) / period;
            avgLoss = (avgLoss * (period - 1) + loss) / period;
            rsi[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
        }
    }
    return rsi;
}

/**
 * ADX - Average Directional Index
 * @param {Object[]} bars - OHLCV bars
 * @param {number} period
 * @returns {{adx: number[], plusDI: number[], minusDI: number[]}}
 */
export function calcADX(bars, period) {
    const len = bars.length;
    const adx = new Array(len).fill(null);
    const plusDI = new Array(len).fill(null);
    const minusDI = new Array(len).fill(null);

    if (len < period * 2) return { adx, plusDI, minusDI };

    // True Range, +DM, -DM raw
    const tr = new Array(len).fill(0);
    const plusDM = new Array(len).fill(0);
    const minusDM = new Array(len).fill(0);

    for (let i = 1; i < len; i++) {
        const high = bars[i].high, low = bars[i].low;
        const prevHigh = bars[i - 1].high, prevLow = bars[i - 1].low, prevClose = bars[i - 1].close;
        tr[i] = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
        const upMove = high - prevHigh;
        const downMove = prevLow - low;
        plusDM[i] = (upMove > downMove && upMove > 0) ? upMove : 0;
        minusDM[i] = (downMove > upMove && downMove > 0) ? downMove : 0;
    }

    // Wilder smoothed sums
    let smoothTR = 0, smoothPlusDM = 0, smoothMinusDM = 0;
    for (let i = 1; i <= period; i++) {
        smoothTR += tr[i];
        smoothPlusDM += plusDM[i];
        smoothMinusDM += minusDM[i];
    }

    let dxSum = 0;
    for (let i = period; i < len; i++) {
        if (i === period) {
            // first smoothed values already computed
        } else {
            smoothTR = smoothTR - smoothTR / period + tr[i];
            smoothPlusDM = smoothPlusDM - smoothPlusDM / period + plusDM[i];
            smoothMinusDM = smoothMinusDM - smoothMinusDM / period + minusDM[i];
        }

        const pdi = smoothTR === 0 ? 0 : (smoothPlusDM / smoothTR) * 100;
        const mdi = smoothTR === 0 ? 0 : (smoothMinusDM / smoothTR) * 100;
        plusDI[i] = pdi;
        minusDI[i] = mdi;

        const diSum = pdi + mdi;
        const dx = diSum === 0 ? 0 : (Math.abs(pdi - mdi) / diSum) * 100;

        if (i < period * 2 - 1) {
            dxSum += dx;
        } else if (i === period * 2 - 1) {
            dxSum += dx;
            adx[i] = dxSum / period;
        } else {
            adx[i] = (adx[i - 1] * (period - 1) + dx) / period;
        }
    }

    return { adx, plusDI, minusDI };
}

/**
 * VWAP with standard deviation bands (daily reset)
 * @param {Object[]} bars - OHLCV bars with .time (unix seconds)
 * @returns {{vwap: number[], upper1: number[], lower1: number[], upper2: number[], lower2: number[]}}
 */
export function calcVWAP(bars) {
    const len = bars.length;
    const vwap = new Array(len).fill(null);
    const upper1 = new Array(len).fill(null);
    const lower1 = new Array(len).fill(null);
    const upper2 = new Array(len).fill(null);
    const lower2 = new Array(len).fill(null);

    let cumTPV = 0;   // cumulative typical-price * volume
    let cumVol = 0;   // cumulative volume
    let cumTPV2 = 0;  // for variance calc: cumulative (tp^2 * vol)
    let prevDay = null;

    for (let i = 0; i < len; i++) {
        const d = new Date(bars[i].time * 1000);
        const day = d.getUTCFullYear() * 10000 + (d.getUTCMonth() + 1) * 100 + d.getUTCDate();

        // Reset on new day
        if (day !== prevDay) {
            cumTPV = 0;
            cumVol = 0;
            cumTPV2 = 0;
            prevDay = day;
        }

        const tp = (bars[i].high + bars[i].low + bars[i].close) / 3;
        const vol = bars[i].volume || 1; // avoid div by zero
        cumTPV += tp * vol;
        cumVol += vol;
        cumTPV2 += tp * tp * vol;

        const v = cumTPV / cumVol;
        vwap[i] = v;

        // Standard deviation of typical price weighted by volume
        const variance = (cumTPV2 / cumVol) - (v * v);
        const sd = Math.sqrt(Math.max(0, variance));

        upper1[i] = v + sd;
        lower1[i] = v - sd;
        upper2[i] = v + 2 * sd;
        lower2[i] = v - 2 * sd;
    }

    return { vwap, upper1, lower1, upper2, lower2 };
}

/**
 * Volume SMA - simple moving average of volume
 * @param {Object[]} bars
 * @param {number} period
 * @returns {number[]}
 */
export function calcVolumeSMA(bars, period) {
    const volumes = bars.map(b => b.volume || 0);
    return calcSMA(volumes, period);
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2: SESSION LEVELS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute session-based levels from bars.
 * Returns levels for the CURRENT day based on PREVIOUS day and overnight data.
 *
 * Sessions (all ET / UTC-5):
 *   RTH: 09:30 - 16:00 ET  (14:30 - 21:00 UTC)
 *   Overnight: 18:00 prev day - 09:30 ET  (23:00 prev UTC - 14:30 UTC)
 *   Opening Range: first 15 min of RTH (09:30 - 09:45 ET)
 *
 * @param {Object[]} bars - OHLCV bars with .time (unix seconds)
 * @returns {{pdh: number|null, pdl: number|null, pdc: number|null, onh: number|null, onl: number|null, orh: number|null, orl: number|null}}
 */
export function calcSessionLevels(bars) {
    if (!bars || bars.length === 0) return { pdh: null, pdl: null, pdc: null, onh: null, onl: null, orh: null, orl: null };

    // Group bars by session
    // RTH = 14:30-21:00 UTC, Overnight = 23:00-14:30 UTC
    const rthDays = {};   // dateKey -> {high, low, close}
    const onSessions = {}; // dateKey -> {high, low}  (overnight leading into that RTH day)
    const orSessions = {}; // dateKey -> {high, low}  (opening range of that RTH day)

    for (const bar of bars) {
        const d = new Date(bar.time * 1000);
        const utcH = d.getUTCHours();
        const utcM = d.getUTCMinutes();
        const utcMinutes = utcH * 60 + utcM;

        // Determine which RTH date this bar belongs to
        // RTH: 14:30-21:00 UTC
        // Overnight before RTH: 23:00 prev day to 14:29 UTC same day
        const dateKey = d.getUTCFullYear() * 10000 + (d.getUTCMonth() + 1) * 100 + d.getUTCDate();

        if (utcMinutes >= 870 && utcMinutes < 1260) {
            // RTH: 14:30 (870min) to 21:00 (1260min) UTC
            if (!rthDays[dateKey]) rthDays[dateKey] = { high: -Infinity, low: Infinity, close: bar.close };
            const s = rthDays[dateKey];
            s.high = Math.max(s.high, bar.high);
            s.low = Math.min(s.low, bar.low);
            s.close = bar.close; // last bar's close = RTH close

            // Opening range: 14:30-14:45 UTC (870-885 min)
            if (utcMinutes >= 870 && utcMinutes < 885) {
                if (!orSessions[dateKey]) orSessions[dateKey] = { high: -Infinity, low: Infinity };
                const o = orSessions[dateKey];
                o.high = Math.max(o.high, bar.high);
                o.low = Math.min(o.low, bar.low);
            }
        } else if (utcMinutes >= 1380 || utcMinutes < 870) {
            // Overnight: 23:00 (1380min) to 14:29 UTC
            // Belongs to the NEXT day's RTH if after 23:00, or same day if before 14:30
            let onDateKey = dateKey;
            if (utcMinutes >= 1380) {
                // After 23:00 UTC = overnight for next calendar day
                const nextDay = new Date(d);
                nextDay.setUTCDate(nextDay.getUTCDate() + 1);
                onDateKey = nextDay.getUTCFullYear() * 10000 + (nextDay.getUTCMonth() + 1) * 100 + nextDay.getUTCDate();
            }
            if (!onSessions[onDateKey]) onSessions[onDateKey] = { high: -Infinity, low: Infinity };
            const s = onSessions[onDateKey];
            s.high = Math.max(s.high, bar.high);
            s.low = Math.min(s.low, bar.low);
        }
    }

    // Get sorted RTH days to find "previous day"
    const sortedDays = Object.keys(rthDays).sort((a, b) => a - b);

    // Current day = last day with data, previous = second to last
    const result = { pdh: null, pdl: null, pdc: null, onh: null, onl: null, orh: null, orl: null };

    if (sortedDays.length >= 2) {
        const prevKey = sortedDays[sortedDays.length - 2];
        const prev = rthDays[prevKey];
        result.pdh = prev.high;
        result.pdl = prev.low;
        result.pdc = prev.close;
    }

    // Overnight for the current/latest day
    const latestDay = sortedDays[sortedDays.length - 1];
    if (latestDay && onSessions[latestDay]) {
        const on = onSessions[latestDay];
        if (on.high !== -Infinity) result.onh = on.high;
        if (on.low !== Infinity) result.onl = on.low;
    }

    // Opening range for current day
    if (latestDay && orSessions[latestDay]) {
        const or = orSessions[latestDay];
        if (or.high !== -Infinity) result.orh = or.high;
        if (or.low !== Infinity) result.orl = or.low;
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 3: REGIME DETECTION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Detect market regime: trending, ranging, or volatile/chaotic
 *
 * Rules:
 *   ADX > 25 → trending
 *   ADX < 20 → ranging
 *   ATR percentile > 80th → volatile/chaotic (overrides above)
 *   Choppiness Index > 61.8 → ranging (confirmation)
 *
 * @param {Object[]} bars
 * @param {{adx: number[], atr: number[]}} indicators - pre-computed
 * @returns {string[]} - regime per bar: 'trending' | 'ranging' | 'volatile'
 */
export function detectRegime(bars, adxValues, atrValues) {
    const len = bars.length;
    const regime = new Array(len).fill('unknown');

    // Compute ATR percentile over rolling 100-bar window
    const atrWindow = 100;

    for (let i = 0; i < len; i++) {
        if (adxValues[i] === null || atrValues[i] === null) continue;

        // ATR percentile
        let atrPercentile = 50;
        if (i >= atrWindow) {
            const window = atrValues.slice(i - atrWindow + 1, i + 1).filter(v => v !== null);
            if (window.length > 0) {
                const sorted = [...window].sort((a, b) => a - b);
                const rank = sorted.indexOf(atrValues[i]);
                atrPercentile = (rank / (sorted.length - 1)) * 100;
            }
        }

        // Choppiness Index (14-period)
        let chop = 50;
        if (i >= 14) {
            const chopPeriod = 14;
            let sumTR = 0;
            let highestHigh = -Infinity;
            let lowestLow = Infinity;
            for (let j = i - chopPeriod + 1; j <= i; j++) {
                const prevClose = j > 0 ? bars[j - 1].close : bars[j].close;
                sumTR += Math.max(bars[j].high - bars[j].low, Math.abs(bars[j].high - prevClose), Math.abs(bars[j].low - prevClose));
                highestHigh = Math.max(highestHigh, bars[j].high);
                lowestLow = Math.min(lowestLow, bars[j].low);
            }
            const range = highestHigh - lowestLow;
            if (range > 0) {
                chop = 100 * Math.log10(sumTR / range) / Math.log10(chopPeriod);
            }
        }

        // Classification
        if (atrPercentile > 80) {
            regime[i] = 'volatile';
        } else if (adxValues[i] > 25) {
            regime[i] = 'trending';
        } else if (adxValues[i] < 20 || chop > 61.8) {
            regime[i] = 'ranging';
        } else {
            regime[i] = 'transition'; // ADX 20-25 zone
        }
    }

    return regime;
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 4: CONFLUENCE SCORING (gradient 0–1)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Score each indicator on a 0-1 gradient scale.
 * Positive = bullish bias, Negative = bearish bias.
 * Returns score in range [-1, 1].
 */

/** VWAP Position Score: where is price relative to VWAP and bands? */
export function scoreVWAP(close, vwap, upper1, lower1, upper2, lower2) {
    if (vwap === null) return 0;
    if (close > upper2) return 1.0;   // extreme above
    if (close > upper1) return 0.7;
    if (close > vwap) return 0.3;
    if (close > lower1) return -0.3;
    if (close > lower2) return -0.7;
    return -1.0; // extreme below
}

/** EMA 9 Score: price relative to EMA as pullback reference */
export function scoreEMA(close, ema) {
    if (ema === null) return 0;
    const dist = (close - ema) / ema; // normalized distance
    // Clamp to [-1, 1] with smooth gradient
    return Math.max(-1, Math.min(1, dist * 100)); // 1% distance = full score
}

/** RSI Score: momentum with 80/20 extremes */
export function scoreRSI(rsi) {
    if (rsi === null) return 0;
    if (rsi >= 80) return 1.0;   // overbought extreme
    if (rsi <= 20) return -1.0;  // oversold extreme
    // Linear scale between 20-80, centered at 50
    return (rsi - 50) / 30; // range [-1, 1]
}

/** ADX Score: trend strength (not direction — magnitude only) */
export function scoreADX(adx) {
    if (adx === null) return 0;
    if (adx > 40) return 1.0;
    if (adx > 25) return (adx - 25) / 15; // 0 to 1
    if (adx > 20) return 0; // transition zone
    return -((20 - adx) / 20); // ranging: negative score
}

/** Volume Score: volume spike relative to SMA */
export function scoreVolume(volume, volSMA) {
    if (volSMA === null || volSMA === 0) return 0;
    const ratio = volume / volSMA;
    if (ratio >= 2.0) return 1.0;  // 2x spike = max
    if (ratio >= 1.5) return 0.7;
    if (ratio >= 1.0) return 0.3;
    return -0.3; // below average = slight negative
}

/**
 * Compute composite confluence score from all sub-scores.
 * Weights: VWAP 0.25, EMA 0.15, RSI 0.20, ADX 0.20, Volume 0.20
 * @returns {number} -1 to 1
 */
export function computeConfluence(vwapScore, emaScore, rsiScore, adxScore, volScore) {
    return vwapScore * 0.25 + emaScore * 0.15 + rsiScore * 0.20 + adxScore * 0.20 + volScore * 0.20;
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 5: DECISION TREE + CONFLICT RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Decision tree: given regime and scores, output action.
 *
 * Hierarchy (highest priority first):
 *   1. Mean Reversion (ranging + RSI extreme + VWAP band touch)
 *   2. Momentum (trending + RSI confirms + volume spike)
 *   3. Trend Following (trending + EMA pullback + ADX strong)
 *   4. Volume Breakout (any regime + 2x volume + price at level)
 *   5. No Trade (conflicting signals or low confluence)
 *
 * @param {string} regime
 * @param {Object} scores - {vwap, ema, rsi, adx, volume, confluence}
 * @param {Object} price - {close, high, low}
 * @param {Object} levels - {pdh, pdl, pdc, onh, onl, orh, orl} or nulls
 * @returns {{action: string, bias: string, confidence: number, reason: string}}
 */
export function decisionTree(regime, scores, price, levels) {
    const { vwap: vwapS, ema: emaS, rsi: rsiS, adx: adxS, volume: volS, confluence } = scores;
    const absConf = Math.abs(confluence);

    // Default
    let result = { action: 'NO_TRADE', bias: 'neutral', confidence: 0, reason: 'Insufficient confluence' };

    // 1. Mean Reversion (ranging regime)
    if (regime === 'ranging' || regime === 'transition') {
        if (rsiS <= -0.7 && vwapS <= -0.5) {
            result = { action: 'BUY', bias: 'bullish', confidence: Math.min(1, absConf + 0.2), reason: 'Mean reversion: oversold at VWAP band' };
        } else if (rsiS >= 0.7 && vwapS >= 0.5) {
            result = { action: 'SELL', bias: 'bearish', confidence: Math.min(1, absConf + 0.2), reason: 'Mean reversion: overbought at VWAP band' };
        }
    }

    // 2. Momentum (trending regime, RSI confirms direction)
    if (result.action === 'NO_TRADE' && regime === 'trending') {
        if (confluence > 0.3 && rsiS > 0.3 && volS > 0.3) {
            result = { action: 'BUY', bias: 'bullish', confidence: absConf, reason: 'Momentum: trend + RSI + volume confirm bullish' };
        } else if (confluence < -0.3 && rsiS < -0.3 && volS > 0.3) {
            result = { action: 'SELL', bias: 'bearish', confidence: absConf, reason: 'Momentum: trend + RSI + volume confirm bearish' };
        }
    }

    // 3. Trend Following (trending + EMA pullback)
    if (result.action === 'NO_TRADE' && regime === 'trending') {
        if (emaS > 0 && emaS < 0.3 && adxS > 0.3) {
            result = { action: 'BUY', bias: 'bullish', confidence: Math.min(1, adxS * 0.8), reason: 'Trend follow: pullback to EMA in uptrend' };
        } else if (emaS < 0 && emaS > -0.3 && adxS > 0.3) {
            result = { action: 'SELL', bias: 'bearish', confidence: Math.min(1, adxS * 0.8), reason: 'Trend follow: pullback to EMA in downtrend' };
        }
    }

    // 4. Volume Breakout (any regime, 2x volume)
    if (result.action === 'NO_TRADE' && volS >= 0.7) {
        // Check proximity to key levels
        const levelProximity = checkLevelProximity(price.close, levels);
        if (levelProximity.near && confluence > 0.2) {
            result = { action: 'BUY', bias: 'bullish', confidence: Math.min(1, volS * 0.7), reason: `Volume breakout near ${levelProximity.level}` };
        } else if (levelProximity.near && confluence < -0.2) {
            result = { action: 'SELL', bias: 'bearish', confidence: Math.min(1, volS * 0.7), reason: `Volume breakdown near ${levelProximity.level}` };
        }
    }

    // 5. Volatile regime override: reduce confidence or block
    if (regime === 'volatile') {
        result.confidence *= 0.5;
        if (result.confidence < 0.3) {
            result = { action: 'NO_TRADE', bias: 'neutral', confidence: 0, reason: 'Volatile regime - signals unreliable' };
        }
    }

    return result;
}

/**
 * Check if price is near any key session level (within 0.15% for futures)
 */
function checkLevelProximity(close, levels) {
    if (!levels) return { near: false, level: '' };
    const threshold = close * 0.0015; // 0.15%
    const checks = [
        ['PDH', levels.pdh], ['PDL', levels.pdl], ['PDC', levels.pdc],
        ['ONH', levels.onh], ['ONL', levels.onl],
        ['ORH', levels.orh], ['ORL', levels.orl],
    ];
    for (const [name, val] of checks) {
        if (val !== null && Math.abs(close - val) < threshold) {
            return { near: true, level: name };
        }
    }
    return { near: false, level: '' };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 8: TIME-OF-DAY SESSION FILTERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if current time is within a tradeable session window.
 *
 * Prime windows (ET):
 *   Morning: 09:30 - 11:30  (first 2 hours of RTH)
 *   Afternoon: 14:00 - 15:45 (last power hour minus close)
 *
 * Avoid:
 *   Lunch: 11:30 - 14:00 (low volume chop)
 *   Last 15 min: 15:45 - 16:00 (erratic close)
 *   Pre-market thin: 04:00 - 08:00 ET (too thin)
 *
 * @param {number} unixSeconds
 * @returns {{tradeable: boolean, session: string, quality: number}}
 */
export function sessionFilter(unixSeconds) {
    const d = new Date(unixSeconds * 1000);
    // Convert UTC to ET (UTC-5, ignoring DST for simplicity — could be improved)
    const etH = (d.getUTCHours() - 5 + 24) % 24;
    const etM = d.getUTCMinutes();
    const etMinutes = etH * 60 + etM;

    // 09:30-11:30 ET = 570-690 min
    if (etMinutes >= 570 && etMinutes < 690) {
        return { tradeable: true, session: 'morning_prime', quality: 1.0 };
    }
    // 14:00-15:45 ET = 840-945 min
    if (etMinutes >= 840 && etMinutes < 945) {
        return { tradeable: true, session: 'afternoon_prime', quality: 0.9 };
    }
    // 08:00-09:30 ET = pre-open
    if (etMinutes >= 480 && etMinutes < 570) {
        return { tradeable: true, session: 'pre_open', quality: 0.5 };
    }
    // 11:30-14:00 ET = lunch
    if (etMinutes >= 690 && etMinutes < 840) {
        return { tradeable: false, session: 'lunch_chop', quality: 0.2 };
    }
    // 15:45-16:00 ET = close
    if (etMinutes >= 945 && etMinutes < 960) {
        return { tradeable: false, session: 'close_erratic', quality: 0.1 };
    }
    // Overnight
    return { tradeable: false, session: 'overnight', quality: 0.3 };
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ENGINE CLASS - Ties everything together
// ═══════════════════════════════════════════════════════════════════════════

export class ScalpingEngine {
    constructor() {
        this.bars = [];
        this.state = null; // latest computed state
    }

    /**
     * Full compute on all bars. Call on init and new bars.
     * @param {Object[]} bars - OHLCV bars
     * @returns {Object} - full state snapshot
     */
    compute(bars) {
        this.bars = bars;
        if (bars.length < 30) return null;

        const closes = bars.map(b => b.close);

        // Phase 1: Core indicators
        const ema9 = calcEMA(closes, 9);
        const rsi7 = calcRSI(closes, 7);
        const { adx: adx10, plusDI, minusDI } = calcADX(bars, 10);
        const atr10 = calcATR(bars, 10);
        const volSMA20 = calcVolumeSMA(bars, 20);
        const { vwap, upper1, lower1, upper2, lower2 } = calcVWAP(bars);

        // Phase 2: Session levels
        const levels = calcSessionLevels(bars);

        // Get latest index
        const i = bars.length - 1;
        const bar = bars[i];

        // Phase 3: Regime
        const regimes = detectRegime(bars, adx10, atr10);
        const regime = regimes[i];

        // Phase 4: Scores
        const vwapScore = scoreVWAP(bar.close, vwap[i], upper1[i], lower1[i], upper2[i], lower2[i]);
        const emaScore = scoreEMA(bar.close, ema9[i]);
        const rsiScore = scoreRSI(rsi7[i]);
        const adxScore = scoreADX(adx10[i]);
        const volScore = scoreVolume(bar.volume, volSMA20[i]);
        const confluence = computeConfluence(vwapScore, emaScore, rsiScore, adxScore, volScore);

        const scores = { vwap: vwapScore, ema: emaScore, rsi: rsiScore, adx: adxScore, volume: volScore, confluence };

        // Phase 5: Decision
        const decision = decisionTree(regime, scores, { close: bar.close, high: bar.high, low: bar.low }, levels);

        // Phase 8: Session filter
        const session = sessionFilter(bar.time);
        if (!session.tradeable && decision.action !== 'NO_TRADE') {
            decision.confidence *= session.quality;
            if (decision.confidence < 0.2) {
                decision.action = 'NO_TRADE';
                decision.reason = `${session.session}: ${decision.reason}`;
            }
        }

        this.state = {
            // Raw indicator values (latest)
            indicators: {
                ema9: ema9[i],
                rsi7: rsi7[i],
                adx10: adx10[i],
                plusDI: plusDI[i],
                minusDI: minusDI[i],
                atr10: atr10[i],
                vwap: vwap[i],
                vwapUpper1: upper1[i],
                vwapLower1: lower1[i],
                vwapUpper2: upper2[i],
                vwapLower2: lower2[i],
                volSMA20: volSMA20[i],
                volume: bar.volume,
            },
            // Full arrays for chart rendering
            series: {
                ema9, rsi7, adx10, plusDI, minusDI, atr10,
                vwap, vwapUpper1: upper1, vwapLower1: lower1, vwapUpper2: upper2, vwapLower2: lower2,
                volSMA20,
            },
            // Levels
            levels,
            // Regime
            regime,
            // Scores
            scores,
            // Decision
            decision,
            // Session
            session,
            // Time
            time: bar.time,
        };

        return this.state;
    }

    /**
     * Quick update for tick (update last bar, recompute only latest values).
     * For performance, only recompute on new bars; ticks just update the last bar.
     */
    updateLastBar(bar) {
        if (this.bars.length === 0) return this.state;
        this.bars[this.bars.length - 1] = bar;
        // Don't recompute on every tick — caller should throttle
        return this.state;
    }

    getState() {
        return this.state;
    }
}
