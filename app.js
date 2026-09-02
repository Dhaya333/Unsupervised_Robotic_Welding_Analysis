/* ============================================================
   WeldScan — Acoustic Quality Monitor
   app.js — Fixed canvas rendering + professional UI logic
   ============================================================ */

let audioContext, analyser, dataArray, freqArray;
let socket, stream, animationId;
let isRunning = false;

/* ===== DOM REFS ===== */
const canvas      = document.getElementById("visualizer");
const specCanvas  = document.getElementById("spectrum");
const ctx         = canvas.getContext("2d");
const specCtx     = specCanvas.getContext("2d");

const audioSelect   = document.getElementById("audioSource");
const startBtn      = document.getElementById("startBtn");
const stopBtn       = document.getElementById("stopBtn");
const timestampEl   = document.getElementById("timestamp");
const connStatusEl  = document.getElementById("connStatus");
const resultBanner  = document.getElementById("resultBanner");
const resultLabel   = document.getElementById("resultLabel");
const resultSub     = document.getElementById("resultSub");
const vizBadge      = document.getElementById("vizBadge");
const logBox        = document.getElementById("logBox");
const historyList   = document.getElementById("historyList");
const peakAmpEl     = document.getElementById("peakAmp");
const rmsLevelEl    = document.getElementById("rmsLevel");
const domFreqEl     = document.getElementById("domFreq");
const zeroCrossEl   = document.getElementById("zeroCross");
const peakMeterEl   = document.getElementById("peakMeter");
const rmsMeterEl    = document.getElementById("rmsMeter");
const verdictEl     = document.getElementById("verdictValue");
const verdictCard   = document.getElementById("verdictCard");

/* ===== CANVAS RESIZE =====
   Canvas pixel size must match CSS display size × devicePixelRatio.
   Drawing coords must use CSS size (logical pixels), not canvas.width.
*/
function resizeCanvases() {
    const dpr = window.devicePixelRatio || 1;
    [canvas, specCanvas].forEach(c => {
        const rect = c.getBoundingClientRect();
        c.width  = Math.round(rect.width  * dpr);
        c.height = Math.round(rect.height * dpr);
    });
}
window.addEventListener("resize", resizeCanvases);
// Defer until layout is complete
requestAnimationFrame(resizeCanvases);

/* ===== CLOCK ===== */
function updateClock() {
    timestampEl.textContent = new Date().toLocaleTimeString("en-GB", { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

/* ===== LOG ===== */
function log(msg, type = "info") {
    const time = new Date().toLocaleTimeString("en-GB", { hour12: false });
    const el = document.createElement("div");
    el.className = `log-entry log-${type}`;
    el.textContent = `[${time}] ${msg}`;
    logBox.appendChild(el);
    logBox.scrollTop = logBox.scrollHeight;
}

/* ===== DEVICE INIT ===== */
async function initializeDevices() {
    try {
        const temp = await navigator.mediaDevices.getUserMedia({ audio: true });
        temp.getTracks().forEach(t => t.stop());
        const devices = await navigator.mediaDevices.enumerateDevices();
        audioSelect.innerHTML = "";
        devices.filter(d => d.kind === "audioinput").forEach(d => {
            const opt = document.createElement("option");
            opt.value = d.deviceId;
            opt.text  = d.label || "Microphone";
            audioSelect.appendChild(opt);
        });
        log("Audio devices loaded.", "info");
    } catch {
        log("Microphone permission denied.", "bad");
    }
}
navigator.mediaDevices.addEventListener("devicechange", initializeDevices);

/* ===== RESULT BANNER ===== */
function setResult(state, label, sub) {
    resultBanner.className = `result-banner ${state}`;
    resultLabel.textContent = label;
    resultSub.textContent   = sub;

    verdictEl.className = "metric-value";
    verdictCard.className = "metric-card verdict-card";

    if (state === "good") {
        verdictEl.textContent = "PASS";
        verdictEl.classList.add("verdict-good");
        verdictCard.classList.add("good");
    } else if (state === "bad") {
        verdictEl.textContent = "FAIL";
        verdictEl.classList.add("verdict-bad");
        verdictCard.classList.add("bad");
    } else {
        verdictEl.textContent = "--";
        verdictEl.classList.add("verdict-idle");
    }
}

/* ===== HISTORY ===== */
function addHistory(result) {
    const empty = historyList.querySelector(".history-empty");
    if (empty) empty.remove();

    const time = new Date().toLocaleTimeString("en-GB", { hour12: false });
    const item = document.createElement("div");
    item.className = `history-item ${result}`;
    item.innerHTML = `<span>${result === "good" ? "PASS — Good Weld" : "FAIL — Bad Weld"}</span>
                      <span class="history-time">${time}</span>`;
    historyList.prepend(item);

    while (historyList.children.length > 30) {
        historyList.removeChild(historyList.lastChild);
    }
}

/* ===== BACKEND ===== */
function connectBackend() {
    socket = io("http://127.0.0.1:5000");

    socket.on("connect", () => {
        connStatusEl.textContent = "ONLINE";
        connStatusEl.className = "meta-value online";
        log("Connected to backend.", "info");
    });

    socket.on("prediction", (data) => {

        if (data.status === "NOT OK") {
            setResult("bad", "BAD WELD", "Acoustic anomaly detected in weld signal.");
            log("Result: Bad weld detected.", "bad");
            addHistory("bad");
        } else if (data.status === "OK") {
            setResult("good", "GOOD WELD", "Weld acoustic signature is within tolerance.");
            log("Result: Good weld confirmed.", "good");
            addHistory("good");
        }

    });

    socket.on("disconnect", () => {
        connStatusEl.textContent = "OFFLINE";
        connStatusEl.className = "meta-value offline";
        log("Disconnected from backend.", "warn");
    });
}
/* ===== ANALYTICS ===== */
function computeMetrics(data, freqData, sampleRate) {
    let peak = 0, sum = 0;
    for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128;
        if (Math.abs(v) > peak) peak = Math.abs(v);
        sum += v * v;
    }
    const rms = Math.sqrt(sum / data.length);

    let maxIdx = 0, maxVal = 0;
    for (let i = 1; i < freqData.length; i++) {
        if (freqData[i] > maxVal) { maxVal = freqData[i]; maxIdx = i; }
    }
    const domFreq = Math.round(maxIdx * sampleRate / analyser.fftSize);

    let zc = 0;
    for (let i = 1; i < data.length; i++) {
        if ((data[i - 1] < 128) !== (data[i] < 128)) zc++;
    }

    return { peak, rms, domFreq, zc };
}

/* ===== DRAW =====*/

function draw() {
    animationId = requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);
    analyser.getByteFrequencyData(freqArray);

    const dpr = window.devicePixelRatio || 1;
    const W  = canvas.width;
    const H  = canvas.height;
    const SW = specCanvas.width;
    const SH = specCanvas.height;

    /* -- Waveform -- */
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, W, H);

    // Subtle grid
    ctx.strokeStyle = "rgba(42, 50, 66, 0.8)";
    ctx.lineWidth = 1;
    // Horizontal lines
    for (let i = 1; i <= 3; i++) {
        const y = Math.round(H * i / 4) + 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
    }
    // Vertical lines
    for (let i = 1; i <= 7; i++) {
        const x = Math.round(W * i / 8) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
    }

    // Centre axis
    ctx.strokeStyle = "rgba(74, 159, 212, 0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.stroke();

    // Waveform line
    ctx.strokeStyle = "#4a9fd4";
    ctx.lineWidth = 1.5 * dpr;
    ctx.lineJoin = "round";
    ctx.beginPath();

    const step = Math.max(1, Math.floor(dataArray.length / W));
    let px = 0;
    for (let i = 0; i < dataArray.length && px < W; i += step) {
        const v = (dataArray[i] - 128) / 128;
        const py = (H / 2) + v * (H * 0.42);
        px === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        px++;
    }
    ctx.stroke();

    /* -- Spectrum -- */
    specCtx.clearRect(0, 0, SW, SH);
    specCtx.fillStyle = "#0d1117";
    specCtx.fillRect(0, 0, SW, SH);

    const numBars = Math.min(256, Math.floor(SW / 3));
    const barW    = SW / numBars;
    const freqStep = Math.floor(freqArray.length / numBars);

    for (let i = 0; i < numBars; i++) {
        const val  = freqArray[i * freqStep] / 255;
        const barH = val * SH;
        // Gradient from blue (low) to lighter blue (high amplitude)
        const alpha = 0.4 + val * 0.6;
        specCtx.fillStyle = `rgba(74, 159, 212, ${alpha})`;
        specCtx.fillRect(i * barW, SH - barH, barW - 1, barH);
    }

    /* -- Metrics -- */
    const { peak, rms, domFreq, zc } = computeMetrics(
        dataArray, freqArray, audioContext.sampleRate
    );

    peakAmpEl.textContent   = peak.toFixed(3);
    rmsLevelEl.textContent  = rms.toFixed(3);
    domFreqEl.innerHTML     = `${domFreq} <span class="unit">Hz</span>`;
    zeroCrossEl.textContent = zc;
    peakMeterEl.style.width = `${Math.min(peak * 100, 100)}%`;
    rmsMeterEl.style.width  = `${Math.min(rms * 200, 100)}%`;
    

    /* -- Send to backend -- */
    if (socket && socket.connected) {
    socket.emit("audio_chunk", dataArray); 
    /*had socket.emit("audio_chunk", dataArray.buffer); */
}
}

/* ===== START ===== */
startBtn.addEventListener("click", async () => {
    if (isRunning) return;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                deviceId: audioSelect.value ? { exact: audioSelect.value } : undefined,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        });
    } catch (err) {
        log("Could not access audio device: " + err.message, "bad");
        return;
    }

    isRunning = true;
    startBtn.disabled = true;
    stopBtn.disabled  = false;

    audioContext = new AudioContext();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 4096;
    analyser.smoothingTimeConstant = 0.85;

    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    dataArray = new Uint8Array(analyser.fftSize);
    freqArray = new Uint8Array(analyser.frequencyBinCount);

    resizeCanvases();
    setResult("analyzing", "ANALYZING", "Processing live audio stream...");
    vizBadge.textContent = "LIVE";
    vizBadge.className   = "viz-badge live";

    log("Analysis started. Sample rate: " + audioContext.sampleRate + " Hz.", "info");

    connectBackend();
    draw();
});

/* ===== STOP ===== */
stopBtn.addEventListener("click", () => {
    if (!isRunning) return;
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled  = true;

    if (animationId) cancelAnimationFrame(animationId);
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (audioContext) audioContext.close();
    if (socket && socket.readyState === WebSocket.OPEN) socket.close();

    // Clear canvases
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    specCtx.clearRect(0, 0, specCanvas.width, specCanvas.height);

    // Reset
    vizBadge.textContent = "STOPPED";
    vizBadge.className   = "viz-badge";
    setResult("idle", "IDLE", "Analysis stopped.");
    peakAmpEl.textContent = "0.000";
    rmsLevelEl.textContent = "0.000";
    domFreqEl.innerHTML = `0 <span class="unit">Hz</span>`;
    zeroCrossEl.textContent = "0";
    peakMeterEl.style.width = "0%";
    rmsMeterEl.style.width  = "0%";

    connStatusEl.textContent = "OFFLINE";
    connStatusEl.className = "meta-value offline";

    log("Analysis stopped.", "info");
});

/* ===== INIT ===== */
initializeDevices();
setResult("idle", "IDLE", "Select an audio source and press Start Analysis");