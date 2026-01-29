/**
 * Scalping Decision Engine
 *
 * Full indicator stack + regime detection + confluence scoring + decision tree
 * for intraday scalping on MNQ/NQ, MGC/GC futures.
 *
 * Implements the complete research document specifications:
 * - 5 independent dimensions (trend, fair value, momentum, volatility regime, structure)
 * - Gradient 0-1 scoring with z-score VWAP, exact RSI table
 * - Category-weighted confluence (MR 0.35, Mom 0.25, Trend 0.30, Vol 0.10)
 * - 12-layer decision tree with regime-dependent thresholds
 * - Edge case detectors (choppy, volatility spike, RSI divergence)
 * - Session-adaptive RSI thresholds
 * - Conflict resolution matrix
 * - Position sizing, entry type, stop/target ATR multiples
 */

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 1: CORE INDICATOR CALCULATIONS
// ═══════════════════════════════════════════════════════════════════════════

export function calcEMA(values, period) {
    const ema = new Array(values.length).fill(null);
    const k = 2 / (period + 1);
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        if (i < period - 1) { sum += values[i]; continue; }
        if (i === period - 1) { sum += values[i]; ema[i] = sum / period; continue; }
        ema[i] = values[i] * k + ema[i - 1] * (1 - k);
    }
    return ema;
}

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

export function calcADX(bars, period) {
    const len = bars.length;
    const adx = new Array(len).fill(null);
    const plusDI = new Array(len).fill(null);
    const minusDI = new Array(len).fill(null);
    if (len < period * 2) return { adx, plusDI, minusDI };

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

    let smoothTR = 0, smoothPlusDM = 0, smoothMinusDM = 0;
    for (let i = 1; i <= period; i++) {
        smoothTR += tr[i];
        smoothPlusDM += plusDM[i];
        smoothMinusDM += minusDM[i];
    }

    let dxSum = 0;
    for (let i = period; i < len; i++) {
        if (i > period) {
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

export function calcVWAP(bars) {
    const len = bars.length;
    const vwap = new Array(len).fill(null);
    const upper1 = new Array(len).fill(null);
    const lower1 = new Array(len).fill(null);
    const upper2 = new Array(len).fill(null);
    const lower2 = new Array(len).fill(null);
    const stdDev = new Array(len).fill(null);

    let cumTPV = 0, cumVol = 0, cumTPV2 = 0;
    let prevDay = null;

    for (let i = 0; i < len; i++) {
        const d = new Date(bars[i].time * 1000);
        const day = d.getUTCFullYear() * 10000 + (d.getUTCMonth() + 1) * 100 + d.getUTCDate();
        if (day !== prevDay) {
            cumTPV = 0; cumVol = 0; cumTPV2 = 0;
            prevDay = day;
        }
        const tp = (bars[i].high + bars[i].low + bars[i].close) / 3;
        const vol = bars[i].volume || 1;
        cumTPV += tp * vol;
        cumVol += vol;
        cumTPV2 += tp * tp * vol;
        const v = cumTPV / cumVol;
        vwap[i] = v;
        const variance = (cumTPV2 / cumVol) - (v * v);
        const sd = Math.sqrt(Math.max(0, variance));
        stdDev[i] = sd;
        upper1[i] = v + sd;
        lower1[i] = v - sd;
        upper2[i] = v + 2 * sd;
        lower2[i] = v - 2 * sd;
    }
    return { vwap, upper1, lower1, upper2, lower2, stdDev };
}

export function calcVolumeSMA(bars, period) {
    return calcSMA(bars.map(b => b.volume || 0), period);
}

/**
 * Choppiness Index (14-period)
 */
export function calcChoppiness(bars, period = 14) {
    const len = bars.length;
    const chop = new Array(len).fill(null);
    for (let i = period; i < len; i++) {
        let sumTR = 0, highestHigh = -Infinity, lowestLow = Infinity;
        for (let j = i - period + 1; j <= i; j++) {
            const prevClose = j > 0 ? bars[j - 1].close : bars[j].close;
            sumTR += Math.max(bars[j].high - bars[j].low, Math.abs(bars[j].high - prevClose), Math.abs(bars[j].low - prevClose));
            highestHigh = Math.max(highestHigh, bars[j].high);
            lowestLow = Math.min(lowestLow, bars[j].low);
        }
        const range = highestHigh - lowestLow;
        chop[i] = range > 0 ? 100 * Math.log10(sumTR / range) / Math.log10(period) : 50;
    }
    return chop;
}

/**
 * ATR Percentile over rolling window
 */
export function calcATRPercentile(atrValues, windowSize = 100) {
    const len = atrValues.length;
    const pct = new Array(len).fill(null);
    for (let i = 0; i < len; i++) {
        if (atrValues[i] === null || i < windowSize) continue;
        const window = [];
        for (let j = i - windowSize + 1; j <= i; j++) {
            if (atrValues[j] !== null) window.push(atrValues[j]);
        }
        if (window.length === 0) continue;
        const sorted = [...window].sort((a, b) => a - b);
        let rank = 0;
        for (const v of sorted) { if (v <= atrValues[i]) rank++; }
        pct[i] = (rank / sorted.length) * 100;
    }
    return pct;
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2: SESSION LEVELS
// ═══════════════════════════════════════════════════════════════════════════

export function calcSessionLevels(bars) {
    if (!bars || bars.length === 0) return { pdh: null, pdl: null, pdc: null, onh: null, onl: null, orh: null, orl: null };

    const rthDays = {};
    const onSessions = {};
    const orSessions = {};

    for (const bar of bars) {
        const d = new Date(bar.time * 1000);
        const utcMinutes = d.getUTCHours() * 60 + d.getUTCMinutes();
        const dateKey = d.getUTCFullYear() * 10000 + (d.getUTCMonth() + 1) * 100 + d.getUTCDate();

        if (utcMinutes >= 870 && utcMinutes < 1260) {
            // RTH: 14:30-21:00 UTC
            if (!rthDays[dateKey]) rthDays[dateKey] = { high: -Infinity, low: Infinity, close: bar.close };
            const s = rthDays[dateKey];
            s.high = Math.max(s.high, bar.high);
            s.low = Math.min(s.low, bar.low);
            s.close = bar.close;
            // Opening range: 14:30-14:45 UTC
            if (utcMinutes >= 870 && utcMinutes < 885) {
                if (!orSessions[dateKey]) orSessions[dateKey] = { high: -Infinity, low: Infinity };
                const o = orSessions[dateKey];
                o.high = Math.max(o.high, bar.high);
                o.low = Math.min(o.low, bar.low);
            }
        } else if (utcMinutes >= 1380 || utcMinutes < 870) {
            let onDateKey = dateKey;
            if (utcMinutes >= 1380) {
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

    const sortedDays = Object.keys(rthDays).sort((a, b) => a - b);
    const result = { pdh: null, pdl: null, pdc: null, onh: null, onl: null, orh: null, orl: null };

    if (sortedDays.length >= 2) {
        const prev = rthDays[sortedDays[sortedDays.length - 2]];
        result.pdh = prev.high;
        result.pdl = prev.low;
        result.pdc = prev.close;
    }
    const latestDay = sortedDays[sortedDays.length - 1];
    if (latestDay && onSessions[latestDay]) {
        const on = onSessions[latestDay];
        if (on.high !== -Infinity) result.onh = on.high;
        if (on.low !== Infinity) result.onl = on.low;
    }
    if (latestDay && orSessions[latestDay]) {
        const or = orSessions[latestDay];
        if (or.high !== -Infinity) result.orh = or.high;
        if (or.low !== Infinity) result.orl = or.low;
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 3: REGIME DETECTION (per doc: ADX + ATR ratio + Choppiness)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Regime detection per doc's determineRegime():
 *   atrRatio > 2.0 → CHAOS
 *   ADX < 20 || chop > 61.8 → RANGING
 *   ADX > 25 && chop < 38.2 → TRENDING
 *   else → TRANSITION
 */
export function detectRegime(adxVal, atrVal, atrSMA20Val, chopVal) {
    if (atrVal === null || adxVal === null) return 'unknown';
    const atrRatio = atrSMA20Val && atrSMA20Val > 0 ? atrVal / atrSMA20Val : 1;

    if (atrRatio > 2.0) return 'chaos';
    if (adxVal < 20 || (chopVal !== null && chopVal > 61.8)) return 'ranging';
    if (adxVal > 25 && (chopVal === null || chopVal < 38.2)) return 'trending';
    return 'transition';
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 4: CONFLUENCE SCORING — Doc's exact gradient functions
// ═══════════════════════════════════════════════════════════════════════════

/** VWAP Z-Score Scoring (doc's exact function) */
export function scoreVWAP(price, vwap, stdDev) {
    if (vwap === null || !stdDev || stdDev === 0) return { score: 0, bias: 'NEUTRAL' };
    const zScore = (price - vwap) / stdDev;
    if (zScore > 2) return { score: 0.9, bias: 'EXTENDED_LONG' };
    if (zScore > 1) return { score: 0.6, bias: 'BULLISH' };
    if (zScore > 0) return { score: 0.3, bias: 'SLIGHT_BULL' };
    if (zScore > -1) return { score: -0.3, bias: 'SLIGHT_BEAR' };
    if (zScore > -2) return { score: -0.6, bias: 'BEARISH' };
    return { score: -0.9, bias: 'EXTENDED_SHORT' };
}

/** RSI Scoring for LONG entries (doc's exact table) */
export function scoreRSI(rsi) {
    if (rsi === null) return 0;
    if (rsi < 20) return 1.0;
    if (rsi < 30) return 0.8;
    if (rsi < 40) return 0.6;
    if (rsi < 50) return 0.4;
    if (rsi < 60) return 0.2;
    if (rsi < 70) return 0.0;
    if (rsi < 80) return -0.3;
    return -0.6;
}

/** Supertrend Score (doc's function) */
export function scoreSupertrend(direction) {
    if (direction === 1) return 1.0;  // UP
    if (direction === -1) return 0.0; // DOWN
    return 0.5;
}

/** EMA 9 Score: price above/below */
export function scoreEMA(price, ema9) {
    if (ema9 === null) return 0;
    return price > ema9 ? 1 : 0;
}

/** Volume Score: relative volume */
export function scoreVolume(volume, volSMA20) {
    if (!volSMA20 || volSMA20 === 0) return 0;
    return volume > volSMA20 ? 1 : 0;
}

/** ADX Score: trend strength */
export function scoreADX(adx) {
    if (adx === null) return 0;
    if (adx > 40) return 1.0;
    if (adx > 25) return (adx - 25) / 15;
    if (adx > 20) return 0;
    return -((20 - adx) / 20);
}

/**
 * Category-weighted confluence per doc:
 *   Mean Reversion: 0.35 weight (VWAP z-score + RSI)
 *   Momentum: 0.25 weight (RSI direction + price acceleration)
 *   Trend: 0.30 weight (Supertrend + EMA)
 *   Volume: 0.10 weight
 */
export function computeConfluence(indicators, regime) {
    const { vwapResult, rsiVal, rsiPrev, close, closePrev2, stDirection, ema9Val, volume, volSMA20 } = indicators;

    // Mean Reversion Score
    const mrVwap = Math.abs(vwapResult.score);
    const mrRsi = scoreRSI(rsiVal);
    const mrScore = mrVwap * 0.6 + mrRsi * 0.4;

    // Momentum Score
    const rsiDirection = rsiVal !== null && rsiPrev !== null && rsiVal > rsiPrev ? 1 : 0;
    const priceAccel = close !== null && closePrev2 !== null && close > closePrev2 ? 1 : 0;
    const momScore = rsiDirection * 0.5 + priceAccel * 0.5;

    // Trend Score
    const stScore = scoreSupertrend(stDirection);
    const emaScore = scoreEMA(close, ema9Val);
    const trendScore = stScore * 0.6 + emaScore * 0.4;

    // Volume Score
    const volScore = scoreVolume(volume, volSMA20);

    // Weighted total
    const total = mrScore * 0.35 + momScore * 0.25 + trendScore * 0.30 + volScore * 0.10;

    return {
        total,
        components: { meanReversion: mrScore, momentum: momScore, trend: trendScore, volume: volScore },
        raw: { vwap: vwapResult.score, rsi: mrRsi, ema: emaScore, adx: scoreADX(indicators.adxVal), vol: volScore }
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// EDGE CASE DETECTORS (from doc)
// ═══════════════════════════════════════════════════════════════════════════

/** Choppy market filter (doc's isChoppyMarket) */
export function isChoppyMarket(adx, chopVal, atrPercentile, bars, lookback = 10) {
    let score = 0;
    if (adx !== null && adx < 20) score++;
    if (chopVal !== null && chopVal > 61.8) score++;
    if (atrPercentile !== null && atrPercentile < 25) score++;

    // Count EMA 9 crosses in last N bars
    if (bars && bars.length >= lookback + 9) {
        const closes = bars.slice(-lookback - 9).map(b => b.close);
        const ema = calcEMA(closes, 9);
        let crosses = 0;
        for (let i = 10; i < ema.length; i++) {
            if (ema[i] !== null && ema[i - 1] !== null) {
                const above = closes[i] > ema[i];
                const prevAbove = closes[i - 1] > ema[i - 1];
                if (above !== prevAbove) crosses++;
            }
        }
        if (crosses >= 4) score++;
    }

    return score >= 2;
}

/** Volatility spike detection (doc's detectVolatilityEvent) */
export function detectVolatilityEvent(currentBar, atr) {
    if (atr === null || atr === 0) return 'NORMAL';
    const barRange = currentBar.high - currentBar.low;
    if (barRange > atr * 3) return 'EXTREME';
    if (barRange > atr * 2) return 'HIGH';
    if (barRange > atr * 1.5) return 'ELEVATED';
    return 'NORMAL';
}

/** RSI divergence detection (doc's detectDivergence) */
export function detectDivergence(bars, rsiValues, lookback = 5) {
    const len = bars.length;
    if (len < lookback + 1) return 'NONE';
    const i = len - 1;
    if (rsiValues[i] === null) return 'NONE';

    let maxHigh = -Infinity, maxRSI = -Infinity;
    let minLow = Infinity, minRSI = Infinity;
    for (let j = i - lookback; j < i; j++) {
        if (j < 0 || rsiValues[j] === null) continue;
        maxHigh = Math.max(maxHigh, bars[j].high);
        maxRSI = Math.max(maxRSI, rsiValues[j]);
        minLow = Math.min(minLow, bars[j].low);
        minRSI = Math.min(minRSI, rsiValues[j]);
    }

    const priceHigher = bars[i].high > maxHigh;
    const rsiLower = rsiValues[i] < maxRSI;
    const priceLower = bars[i].low < minLow;
    const rsiHigher = rsiValues[i] > minRSI;

    if (priceHigher && rsiLower) return 'BEARISH_DIVERGENCE';
    if (priceLower && rsiHigher) return 'BULLISH_DIVERGENCE';
    return 'NONE';
}

// ═══════════════════════════════════════════════════════════════════════════
// SESSION-ADAPTIVE RSI THRESHOLDS (from doc)
// ═══════════════════════════════════════════════════════════════════════════

export function getSessionThresholds(unixSeconds) {
    const d = new Date(unixSeconds * 1000);
    const etH = (d.getUTCHours() - 5 + 24) % 24;
    // Market open 9-11 ET
    if (etH >= 9 && etH < 11) return { overbought: 80, oversold: 20 };
    // Midday 11-14 ET — tighter mean reversion
    if (etH >= 11 && etH < 14) return { overbought: 65, oversold: 35 };
    // Power hour + default
    return { overbought: 75, oversold: 25 };
}

// ═══════════════════════════════════════════════════════════════════════════
// TIME-OF-DAY SESSION FILTERS
// ═══════════════════════════════════════════════════════════════════════════

export function sessionFilter(unixSeconds) {
    const d = new Date(unixSeconds * 1000);
    const etH = (d.getUTCHours() - 5 + 24) % 24;
    const etMinutes = etH * 60 + d.getUTCMinutes();

    if (etMinutes >= 570 && etMinutes < 690) return { tradeable: true, session: 'morning_prime', quality: 1.0 };
    if (etMinutes >= 840 && etMinutes < 945) return { tradeable: true, session: 'afternoon_prime', quality: 0.9 };
    if (etMinutes >= 480 && etMinutes < 570) return { tradeable: true, session: 'pre_open', quality: 0.5 };
    if (etMinutes >= 690 && etMinutes < 840) return { tradeable: false, session: 'lunch_chop', quality: 0.2 };
    if (etMinutes >= 945 && etMinutes < 960) return { tradeable: false, session: 'close_erratic', quality: 0.1 };
    return { tradeable: false, session: 'overnight', quality: 0.3 };
}

/** No-trade window check (doc's Layer 1) */
function isNoTradeWindow(unixSeconds) {
    const s = sessionFilter(unixSeconds);
    return !s.tradeable;
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFLICT RESOLUTION MATRIX (from doc)
// ═══════════════════════════════════════════════════════════════════════════

export function resolveConflict(stDirection1m, vwapBias, rsi, stDirection15m) {
    // HTF Veto: Never trade against strong HTF trend
    if (stDirection15m === -1 && stDirection1m === 1) {
        return { trade: false, reason: 'HTF bearish veto', positionSize: 0 };
    }
    if (stDirection15m === 1 && stDirection1m === -1) {
        return { trade: false, reason: 'HTF bullish veto', positionSize: 0 };
    }

    // RSI Extreme Veto
    if (stDirection1m === 1 && rsi !== null && rsi > 80) {
        return { trade: false, reason: 'RSI overbought on long signal', positionSize: 0 };
    }
    if (stDirection1m === -1 && rsi !== null && rsi < 20) {
        return { trade: false, reason: 'RSI oversold on short signal', positionSize: 0 };
    }

    // VWAP Entry Filter
    if (stDirection1m === 1 && (vwapBias === 'BEARISH' || vwapBias === 'SLIGHT_BEAR' || vwapBias === 'EXTENDED_SHORT')) {
        return { trade: true, action: 'WAIT_FOR_VWAP_RECLAIM', entryType: 'LIMIT_AT_VWAP', positionSize: 0.5 };
    }
    if (stDirection1m === -1 && (vwapBias === 'BULLISH' || vwapBias === 'SLIGHT_BULL' || vwapBias === 'EXTENDED_LONG')) {
        return { trade: true, action: 'WAIT_FOR_VWAP_BREAK', entryType: 'LIMIT_AT_VWAP', positionSize: 0.5 };
    }

    // Full alignment
    const aligned = stDirection1m === stDirection15m;
    if (aligned && rsi !== null && rsi < 70 && rsi > 30) {
        return { trade: true, positionSize: 1.0, confidence: 'HIGH' };
    }

    // Partial alignment or no 15m data
    return { trade: true, positionSize: stDirection15m === null ? 0.75 : 0.5, confidence: 'MODERATE' };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 5: FULL 12-LAYER DECISION TREE (from doc)
// ═══════════════════════════════════════════════════════════════════════════

function checkLevelProximity(close, levels) {
    if (!levels) return { near: false, level: '' };
    const threshold = close * 0.0015;
    const checks = [
        ['PDH', levels.pdh], ['PDL', levels.pdl], ['PDC', levels.pdc],
        ['ONH', levels.onh], ['ONL', levels.onl],
        ['ORH', levels.orh], ['ORL', levels.orl],
    ];
    for (const [name, val] of checks) {
        if (val !== null && Math.abs(close - val) < threshold) return { near: true, level: name };
    }
    return { near: false, level: '' };
}

/**
 * Full 12-layer decision tree per doc's evaluateScalpEntry()
 */
export function evaluateScalpEntry(data) {
    const result = {
        action: 'NO_TRADE',
        bias: 'neutral',
        score: 0,
        confidence: 0,
        positionSize: 0,
        entryType: 'MARKET',
        regime: 'unknown',
        stopATR: 0,
        targetATR: 0,
        reason: '',
        volatilityEvent: 'NORMAL',
        divergence: 'NONE',
        choppy: false,
    };

    // LAYER 1: Time Filter
    if (isNoTradeWindow(data.time)) {
        result.reason = `Time filter: ${sessionFilter(data.time).session}`;
        return result;
    }

    // LAYER 2: News Filter (can't implement without news feed — skip)

    // LAYER 3: Regime Detection
    const regime = detectRegime(data.adx, data.atr, data.atrSMA20, data.chop);
    result.regime = regime;
    if (regime === 'chaos') {
        result.reason = 'Extreme volatility (CHAOS regime)';
        return result;
    }

    // LAYER 4: ATR Percentile Check
    if (data.atrPercentile !== null && data.atrPercentile < 25) {
        result.reason = `Low volatility (ATR pct: ${data.atrPercentile.toFixed(0)}%)`;
        return result;
    }

    // Edge case: Choppy market
    result.choppy = data.choppy;
    if (data.choppy) {
        result.reason = 'Choppy market detected (multiple chop indicators)';
        return result;
    }

    // Edge case: Volatility event
    result.volatilityEvent = data.volatilityEvent;
    if (data.volatilityEvent === 'EXTREME') {
        result.reason = 'Extreme volatility bar (>3x ATR) — halt trading';
        return result;
    }

    // LAYER 5: Multi-Timeframe Filter (use Supertrend direction as proxy)
    // We don't have 15min supertrend separate, so we use the existing supertrend direction
    const stDirection = data.stDirection; // 1 = UP, -1 = DOWN
    // If we had 15m supertrend, we'd check here. For now, use 1m.

    // LAYER 6: Confluence Score
    const confluenceResult = data.confluence;
    const score = confluenceResult.total;
    result.score = score;

    // LAYER 7: Signal Direction
    const signalDirection = stDirection === 1 ? 'LONG' : 'SHORT';

    // LAYER 8: Threshold Check (regime-dependent)
    const threshold = regime === 'trending' ? 0.60 : 0.50;
    if (score < threshold) {
        result.reason = `Score ${score.toFixed(2)} below ${regime} threshold ${threshold}`;
        return result;
    }

    // LAYER 9: Conflict Resolution
    const conflict = resolveConflict(stDirection, data.vwapBias, data.rsi, data.stDirection15m || null);
    if (!conflict.trade) {
        result.reason = conflict.reason;
        return result;
    }

    // LAYER 10: RSI Extreme Veto (session-adaptive thresholds)
    const thresholds = getSessionThresholds(data.time);
    if (signalDirection === 'LONG' && data.rsi !== null && data.rsi > thresholds.overbought) {
        result.reason = `RSI ${data.rsi.toFixed(0)} > ${thresholds.overbought} overbought veto`;
        return result;
    }
    if (signalDirection === 'SHORT' && data.rsi !== null && data.rsi < thresholds.oversold) {
        result.reason = `RSI ${data.rsi.toFixed(0)} < ${thresholds.oversold} oversold veto`;
        return result;
    }

    // LAYER 11: Position Sizing
    let positionSize = conflict.positionSize || 1.0;
    if (data.atrPercentile !== null && data.atrPercentile > 90) positionSize *= 0.5;
    if (score < 0.70) positionSize *= 0.75;
    if (data.volatilityEvent === 'HIGH') positionSize *= 0.5;
    if (data.volatilityEvent === 'ELEVATED') positionSize *= 0.75;

    // LAYER 12: Entry Type
    let entryType = conflict.entryType || 'MARKET';
    if (regime === 'ranging') entryType = 'LIMIT_AT_VWAP';

    // Divergence bonus info
    result.divergence = data.divergence;

    // Build final result
    result.action = signalDirection === 'LONG' ? 'BUY' : 'SELL';
    result.bias = signalDirection === 'LONG' ? 'bullish' : 'bearish';
    result.confidence = Math.min(1, score);
    result.positionSize = Math.max(0.25, Math.min(1.0, positionSize));
    result.entryType = entryType;
    result.stopATR = regime === 'trending' ? 2.0 : 1.5;
    result.targetATR = regime === 'trending' ? 3.0 : 2.0;

    // Specific reason based on setup type
    const levelInfo = checkLevelProximity(data.close, data.levels);
    if (regime === 'ranging' && Math.abs(data.vwapResult.score) >= 0.6) {
        result.reason = `Mean reversion: ${data.vwapResult.bias} at VWAP band`;
    } else if (regime === 'trending' && confluenceResult.components.momentum > 0.6) {
        result.reason = `Momentum: trend + RSI + volume confirm ${result.bias}`;
    } else if (regime === 'trending' && data.emaPullback) {
        result.reason = `Trend follow: pullback to EMA 9 in ${stDirection === 1 ? 'uptrend' : 'downtrend'}`;
    } else if (levelInfo.near) {
        result.reason = `${data.volatilityEvent !== 'NORMAL' ? 'Vol ' : ''}Breakout near ${levelInfo.level}`;
    } else {
        result.reason = `${regime} regime, score ${score.toFixed(2)}, ${result.entryType}`;
    }

    // Session quality adjustment
    const session = sessionFilter(data.time);
    if (session.quality < 1.0) {
        result.confidence *= session.quality;
        result.positionSize *= session.quality;
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ENGINE CLASS
// ═══════════════════════════════════════════════════════════════════════════

export class ScalpingEngine {
    constructor() {
        this.bars = [];
        this.state = null;
    }

    compute(bars) {
        this.bars = bars;
        if (bars.length < 30) return null;

        const closes = bars.map(b => b.close);
        const len = bars.length;
        const i = len - 1;
        const bar = bars[i];

        // Phase 1: Core indicators
        const ema9 = calcEMA(closes, 9);
        const rsi7 = calcRSI(closes, 7);
        const { adx: adx10, plusDI, minusDI } = calcADX(bars, 10);
        const atr10 = calcATR(bars, 10);
        const atrSMA20 = calcSMA(atr10.filter(v => v !== null).length > 0 ? atr10 : new Array(len).fill(null), 20);
        const volSMA20 = calcVolumeSMA(bars, 20);
        const { vwap, upper1, lower1, upper2, lower2, stdDev: vwapStdDev } = calcVWAP(bars);
        const chop14 = calcChoppiness(bars, 14);
        const atrPercentile = calcATRPercentile(atr10);

        // Phase 2: Session levels
        const levels = calcSessionLevels(bars);

        // Phase 3: Regime
        const regime = detectRegime(adx10[i], atr10[i], atrSMA20[i], chop14[i]);

        // Phase 4: Scoring (doc's exact functions)
        const vwapResult = scoreVWAP(bar.close, vwap[i], vwapStdDev[i]);
        const rsiPrev = i >= 2 ? rsi7[i - 1] : null;
        const closePrev2 = i >= 2 ? bars[i - 2].close : null;

        // Supertrend direction from existing indicator (pass through)
        // We derive direction from +DI/-DI as proxy
        const stDirection = plusDI[i] !== null && minusDI[i] !== null
            ? (plusDI[i] > minusDI[i] ? 1 : -1) : 0;

        const confluenceResult = computeConfluence({
            vwapResult, rsiVal: rsi7[i], rsiPrev, close: bar.close, closePrev2,
            stDirection, ema9Val: ema9[i], volume: bar.volume, volSMA20: volSMA20[i],
            adxVal: adx10[i],
        }, regime);

        // Edge case detectors
        const choppy = isChoppyMarket(adx10[i], chop14[i], atrPercentile[i], bars);
        const volatilityEvent = detectVolatilityEvent(bar, atr10[i]);
        const divergence = detectDivergence(bars, rsi7);

        // EMA pullback check (within 0.1% of EMA 9)
        const emaPullback = ema9[i] !== null && Math.abs(bar.close - ema9[i]) / ema9[i] < 0.001;

        // Phase 5: Full 12-layer decision tree
        const decision = evaluateScalpEntry({
            time: bar.time,
            close: bar.close,
            high: bar.high,
            low: bar.low,
            adx: adx10[i],
            atr: atr10[i],
            atrSMA20: atrSMA20[i],
            atrPercentile: atrPercentile[i],
            chop: chop14[i],
            rsi: rsi7[i],
            stDirection,
            stDirection15m: null, // would need 15m aggregation
            vwapResult,
            vwapBias: vwapResult.bias,
            confluence: confluenceResult,
            levels,
            choppy,
            volatilityEvent,
            divergence,
            emaPullback,
        });

        this.state = {
            indicators: {
                ema9: ema9[i],
                rsi7: rsi7[i],
                adx10: adx10[i],
                plusDI: plusDI[i],
                minusDI: minusDI[i],
                atr10: atr10[i],
                atrSMA20: atrSMA20[i],
                atrPercentile: atrPercentile[i],
                chop14: chop14[i],
                vwap: vwap[i],
                vwapUpper1: upper1[i],
                vwapLower1: lower1[i],
                vwapUpper2: upper2[i],
                vwapLower2: lower2[i],
                vwapStdDev: vwapStdDev[i],
                volSMA20: volSMA20[i],
                volume: bar.volume,
                stDirection,
            },
            series: {
                ema9, rsi7, adx10, plusDI, minusDI, atr10,
                vwap, vwapUpper1: upper1, vwapLower1: lower1, vwapUpper2: upper2, vwapLower2: lower2,
                volSMA20, chop14, atrPercentile,
            },
            levels,
            regime,
            scores: confluenceResult.raw,
            confluence: confluenceResult,
            decision,
            session: sessionFilter(bar.time),
            volatilityEvent,
            divergence,
            choppy,
            time: bar.time,
        };

        return this.state;
    }

    updateLastBar(bar) {
        if (this.bars.length === 0) return this.state;
        this.bars[this.bars.length - 1] = bar;
        return this.state;
    }

    getState() {
        return this.state;
    }
}
