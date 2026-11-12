// Navigation and practice
function showPage(page) {
    if (page !== "learn") stopCamera();
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    document.getElementById(page).classList.add("active");
    document.querySelectorAll("nav [data-page]").forEach(button => {
        if (button.dataset.page === page) button.setAttribute("aria-current", "page");
        else button.removeAttribute("aria-current");
    });
}

function openPractice(category = "Alphabet") {
    showPage("learn");
    const practice = document.getElementById("practice");
    practice.classList.remove("hidden");
    document.getElementById("practiceTitle").textContent = isNumberReference() ? "Number Reference" : "Alphabet Practice";
    if (!["Alphabet", "Practice"].includes(category)) {
        showFeedback("Alphabet recognition is available now. Vocabulary and sentences are coming later.");
    }
    practice.scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
}

function closePractice() {
    stopCamera();
    document.getElementById("practice").classList.add("hidden");
}

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [17, 0],
];
// The launcher serves the website and LSTM API together on port 5011.
const localPreview = location.protocol === "file:" ||
    ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
const API = window.SIGNBRIDGE_API_URL || (localPreview ? "http://127.0.0.1:5011" : location.origin);

async function readModelJSON(response) {
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.toLowerCase().includes("application/json")) {
        throw new Error("This address returned a webpage instead of the ComVision API response. Check the configured ComVision API URL.");
    }
    try {
        return await response.json();
    } catch {
        throw new Error("The ComVision server returned an unreadable response. Restart the ComVision API and try again.");
    }
}
let session = null;
let lastCandidate = "";
let repeatCount = 0;

function showFeedback(message, state = "") {
    const feedback = document.getElementById("feedback");
    feedback.textContent = message;
    feedback.className = "feedback" + (state ? " " + state : "");
}

function resetPrediction(message = "—") {
    lastCandidate = "";
    repeatCount = 0;
    document.getElementById("gesture").textContent = message;
    document.getElementById("confidence").textContent = "0%";
    document.getElementById("progressBar").style.width = "0%";
}

function drawServerLandmarks(landmarks) {
    if (!Array.isArray(landmarks) || landmarks.length !== 21 ||
        !landmarks.every(point => Array.isArray(point) && point.length === 2 &&
            point.every(Number.isFinite))) {
        throw new Error("Invalid landmark response");
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const points = landmarks.map(([x, y]) => [x * canvas.width, y * canvas.height]);
    ctx.strokeStyle = "#8b5cf6";
    ctx.lineWidth = Math.max(3, canvas.width / 320);
    for (const [from, to] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(points[from][0], points[from][1]);
        ctx.lineTo(points[to][0], points[to][1]);
        ctx.stroke();
    }
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = Math.max(1, canvas.width / 640);
    for (const [x, y] of points) {
        ctx.beginPath();
        ctx.arc(x, y, Math.max(4, canvas.width / 160), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
}

// One explicit recording, followed by one prediction. Camera preview stays live.
const recordButton = document.createElement("button");
recordButton.id = "recordSign";
recordButton.type = "button";
recordButton.className = document.getElementById("startCamera").className;
recordButton.textContent = "Record sign (2 seconds)";
recordButton.disabled = true;
recordButton.style.margin = "8px";
document.getElementById("startCamera").insertAdjacentElement("afterend", recordButton);
const recordStatus = document.createElement("p");
recordStatus.id = "recordStatus";
recordStatus.setAttribute("role", "status");
recordStatus.setAttribute("aria-live", "polite");
recordStatus.textContent = "Enable the camera, then record one complete sign.";
recordButton.insertAdjacentElement("afterend", recordStatus);
const recordProgress = document.createElement("progress");
recordProgress.max = 2;
recordProgress.value = 0;
recordProgress.setAttribute("aria-label", "Recording progress");
recordProgress.style.width = "100%";
recordStatus.insertAdjacentElement("afterend", recordProgress);
const guesses = document.createElement("p");
guesses.id = "sequenceGuesses";
recordProgress.insertAdjacentElement("afterend", guesses);

// Keep only the most recent attempt in memory. Export is always user initiated.
let lastReviewAttempt = null;
const saveAttemptButton = document.createElement("button");
saveAttemptButton.id = "saveAttempt";
saveAttemptButton.type = "button";
saveAttemptButton.className = "secondary";
saveAttemptButton.textContent = "Save attempt for review";
saveAttemptButton.hidden = true;
saveAttemptButton.style.order = "4";
guesses.insertAdjacentElement("afterend", saveAttemptButton);
saveAttemptButton.addEventListener("click", () => {
    if (!lastReviewAttempt) return;
    const blob = new Blob([JSON.stringify(lastReviewAttempt)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "signbridge-attempt-" + lastReviewAttempt.reference_letter + "-" +
        lastReviewAttempt.created_at.replace(/[:.]/g, "-") + ".json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showFeedback("Review file download started. It contains this camera recording and prediction; share it only if you choose.");
});


function refreshRecordButton() {
    recordButton.disabled = !session?.stream || session.busy || isNumberReference();
}
function captureHand() {
    if (!video.videoWidth || !video.videoHeight) return null;
    const frame = document.createElement("canvas");
    const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
    frame.width = Math.round(video.videoWidth * scale);
    frame.height = Math.round(video.videoHeight * scale);
    // Send unmirrored pixels: the Python reader mirrors exactly as in training.
    frame.getContext("2d").drawImage(video, 0, 0, frame.width, frame.height);
    return frame.toDataURL("image/jpeg", 0.85);
}
async function recordSign() {
    const current = session;
    if (!current?.stream || current.busy || isNumberReference()) return;
    const version = current.handVersion;
    const valid = () => session === current && current.handVersion === version;
    current.busy = true;
    lastReviewAttempt = null;
    saveAttemptButton.hidden = true;
    document.body.classList.add("recording");
    refreshRecordButton();
    resetPrediction();
    guesses.textContent = "";
    recordProgress.value = 0;
    const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
    let timer;
    try {
        for (let n = 3; n > 0; n--) {
            if (!valid()) return;
            recordStatus.textContent = "Get ready: " + n;
            recordButton.textContent = "Get ready…";
            showFeedback("Prepare your hand. Start signing when you see GO.");
            await pause(1000);
        }
        if (!valid()) return;
        recordButton.textContent = "Recording…";
        const frames = [], timestamps = [];
        const started = performance.now();
        while (performance.now() - started < 2000) {
            if (!valid()) return;
            const tick = performance.now();
            const elapsed = (tick - started) / 1000;
            const image = captureHand();
            if (!image) throw new Error("Camera is not ready. Please try again.");
            frames.push(image); timestamps.push(elapsed);
            recordProgress.value = elapsed;
            recordStatus.textContent = "● GO — recording: " + (2 - elapsed).toFixed(1) + " seconds left";
            showFeedback("Perform one complete sign. Keep the same hand visible.");
            await pause(Math.max(0, 1000 / 15 - (performance.now() - tick)));
        }
        if (!valid()) return;
        recordProgress.value = 2;
        recordStatus.textContent = "Recording finished. Predicting…";
        recordButton.textContent = "Predicting…";
        const controller = new AbortController();
        current.predictionController = controller;
        timer = setTimeout(() => controller.abort(), 60000);
        const side = document.getElementById("practiceHand").value;
        const payload = {frames, timestamps, hand: side === "left" ? "Left" : "Right"};
        lastReviewAttempt = {
            format: "signbridge-review-v1", created_at: new Date().toISOString(),
            reference_letter: referenceSelect.value, ...payload,
            mirrored_pixels: false,
            camera: {width: video.videoWidth, height: video.videoHeight},
            result: null
        };
        const response = await fetch(API + "/predict-sequence", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
            signal: controller.signal
        });
        const result = await readModelJSON(response);
        if (!valid()) return;
        lastReviewAttempt.result = result;
        lastReviewAttempt.http_status = response.status;
        saveAttemptButton.hidden = false;
        if (!response.ok) throw new Error(result.error || "Prediction failed");
        if (!/^[A-Z]$/.test(result.letter) || !Number.isFinite(result.confidence)
            || result.confidence < 0 || result.confidence > 1) throw new Error("Invalid model response");
        document.getElementById("gesture").textContent = result.letter;
        const percent = Math.round(result.confidence * 100);
        document.getElementById("confidence").textContent = percent + "%";
        document.getElementById("progressBar").style.width = percent + "%";
        document.getElementById("detectedHand").textContent = result.hand + " hand";
        recordStatus.textContent = "Complete — " + result.tracked_frames + "/" + result.total_frames + " frames tracked";
        guesses.textContent = "Top guesses: " + result.top_predictions.map(x => x.letter + " " + (x.confidence * 100).toFixed(1) + "%").join(" · ");
        showFeedback("Predicted " + result.letter + ". " + (result.confidence < 0.7 ? "Low confidence — try another recording. " : "") + "Confidence is not measured accuracy.");
    } catch (error) {
        if (!valid()) return;
        resetPrediction();
        recordStatus.textContent = "Recording could not be predicted.";
        showFeedback(error.name === "AbortError" ? "Prediction timed out. Check the API and try again." : error.message);
    } finally {
        clearTimeout(timer);
        current.busy = false;
        document.body.classList.remove("recording");
        current.predictionController = null;
        recordButton.textContent = "Record sign (2 seconds)";
        refreshRecordButton();
    }
}
recordButton.addEventListener("click", recordSign);

async function startCamera() {
    if (session) return;
    const current = {busy: false, hasHand: false, handVersion: 0, lastRequest: -Infinity,
        stream: null, raf: null, controller: new AbortController()};
    session = current;
    const start = document.getElementById("startCamera");
    start.disabled = true;
    start.textContent = "Starting…";
    resetPrediction();
    showFeedback("Connecting to the model…");
    try {
        const timeout = setTimeout(() => current.controller.abort(), 10000);
        let response;
        try { response = await fetch(API + "/health", {signal: current.controller.signal}); }
        finally { clearTimeout(timeout); }
        const health = await readModelJSON(response);
        if (!response.ok || health.status !== "ready" || health.model !== "lstm_classifier.keras"
            || health.mode !== "sequence") throw new Error("Start the ComVision API on port 5011.");
        if (session !== current) return;
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("Open the website on localhost or HTTPS to use the camera.");
        showFeedback("Allow camera access to start practicing.");
        const stream = await navigator.mediaDevices.getUserMedia({video: {width: 960, height: 720}, audio: false});
        if (session !== current) { stream.getTracks().forEach(track => track.stop()); return; }
        current.stream = stream;
        video.srcObject = stream;
        await video.play();
        if (session !== current) return;
        document.getElementById("cameraPlaceholder").style.display = "none";
        document.getElementById("modelStatus").textContent = "A–Z LSTM connected";
        start.textContent = "Camera running";
        showFeedback("Keep one hand clearly in view…");
        refreshRecordButton();
        recordStatus.textContent = "Ready. Press Record sign for a two-second attempt.";
        showFeedback("Press Record sign. Hold static letters; perform the full movement for J and Z.");
    } catch (error) {
        if (session !== current) return;
        stopCamera();
        document.getElementById("modelStatus").textContent = "Not connected";
        const cameraErrors = {NotAllowedError: "Camera permission was denied. Allow it in your browser, then try again.",
            NotFoundError: "No camera was found. Connect a webcam and try again.",
            NotReadableError: "The camera is busy. Close other camera apps and try again."};
        showFeedback(cameraErrors[error.name] || (error.name === "TypeError" || error.name === "AbortError"
            ? "Start the ComVision API and serve the website over localhost or HTTPS." : error.message));
        console.error(error);
    }
}

function stopCamera() {
    document.body.classList.remove("recording");
    document.getElementById("detectedHand").textContent = "Camera off";
    const previous = session;
    session = null;
    if (previous) {
        previous.controller.abort();
        previous.predictionController?.abort();
        cancelAnimationFrame(previous.raf);
        previous.stream?.getTracks().forEach(track => track.stop());
    }
    recordButton.disabled = true;
    recordProgress.value = 0;
    recordStatus.textContent = "Camera off. Enable the camera to record.";
    guesses.textContent = "";
    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("cameraPlaceholder").style.display = "flex";
    const start = document.getElementById("startCamera");
    start.disabled = false;
    start.textContent = "Enable Camera";
    resetPrediction();
    showFeedback("Enable the camera to practice A–Z.");
}

window.addEventListener("pagehide", stopCamera);
document.addEventListener("visibilitychange", () => { if (document.hidden) stopCamera(); });


// Photo references use the supplied chart unchanged, shown through CSS viewports.
const alphabet = [..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"];
function photoMarkup(letter) {
    const i = alphabet.indexOf(letter);
    const x = (i % 7) / 6 * 100;
    const y = (45 + Math.floor(i / 7) * 137) / (635 - 135) * 100;
    return `<div class="photo-sign" style="background-position:${x}% ${y}%" role="img" aria-label="ASL letter ${letter}, photographic reference"></div>`;
}
function displaySigns(items = alphabet) {
    document.getElementById("signList").innerHTML = items.length ? items.map(letter =>
        `<button class="alphabet-card" onclick="beginLetter('${letter}')" aria-label="Practice letter ${letter}">${photoMarkup(letter)}<span class="card-caption">Letter ${letter}<span>${['J','Z'].includes(letter) ? 'Motion ↗' : 'Practice →'}</span></span></button>`).join("")
        : '<p class="empty-state">No matching letter. Try a letter from A to Z.</p>';
}
function searchSigns() {
    const query = document.getElementById("search").value.trim().toUpperCase();
    displaySigns(alphabet.filter(letter => !query || letter === query || ("LETTER " + letter) === query));
}
function beginLetter(letter) {
    referenceSelect.value = letter;
    updateSignReference();
    openPractice();
}
function randomPractice() { beginLetter(alphabet[Math.floor(Math.random() * alphabet.length)]); }
displaySigns();

// Reference selection is independent of the model's predicted letter.
const referenceSelect = document.getElementById("referenceSign");
function isNumberReference() { return /^[0-9]$/.test(referenceSelect.value); }
function showNumberReference() {
    resetPrediction("Reference only");
    showFeedback("Copy the number sign shown above. Number recognition is not trained yet.");
}
function updateSignReference() {
    const sign = referenceSelect.value;
    document.querySelectorAll(".sign-reference-image").forEach(image => {
        image.hidden = image.dataset.sign !== sign;
        // SVG does not consistently expose the HTML hidden property.
        image.toggleAttribute("hidden", image.dataset.sign !== sign);
    });
    const number = isNumberReference();
    const motion = ["J", "Z"].includes(sign);
    document.getElementById("referenceInstruction").textContent = motion
        ? "Follow the arrow to draw " + sign + " with your " + (sign === "J" ? "little finger." : "index finger.")
        : "Match the hand shape, then hold it steady.";
    const limit = document.getElementById("referenceLimit");
    limit.hidden = !number && !motion;
    limit.textContent = number ? "Reference only: the current model recognizes A–Z, not numbers."
        : motion ? "Record the complete movement during the two-second GO period." : "";
    document.getElementById("practiceTitle").textContent = number ? "Number Reference" : "Alphabet Practice";
    if (session) {
        session.handVersion++;
        session.predictionController?.abort();
    }
    refreshRecordButton();
    recordStatus.textContent = session?.stream ? "Press Record sign when ready." : "Enable your camera, then record your sign.";
    resetPrediction();
    if (number) showNumberReference();
    else showFeedback(session ? "Try the " + sign + " sign shown above." : "Enable the camera to practice " + sign + ".");
}
function stepSign(direction) {
    const options = referenceSelect.options;
    referenceSelect.selectedIndex = (referenceSelect.selectedIndex + direction + options.length) % options.length;
    updateSignReference();
}
referenceSelect.addEventListener("change", updateSignReference);
document.getElementById("previousSign").addEventListener("click", () => stepSign(-1));
document.getElementById("nextSign").addEventListener("click", () => stepSign(1));
updateSignReference();

document.getElementById("practiceHand").addEventListener("change", () => {
    document.getElementById("detectedHand").textContent = session ? "Checking…" : "Camera off";
    if (session) { session.hasHand = false; session.sideFilter = null; }
    // Discard any prediction made with the previous orientation.
    if (session) {
        session.handVersion++;
        session.predictionController?.abort();
    }
    resetPrediction();
    recordStatus.textContent = "Hand selection changed. Press Record sign when ready.";
    const left = document.getElementById("practiceHand").value === "left";
    document.getElementById("handModeNote").textContent = left
        ? "Show your left hand. Recognition pauses if the other hand is detected."
        : "Show your right hand. Recognition pauses if the other hand is detected.";
    if (isNumberReference()) showNumberReference();
    else showFeedback(session ? "Show your " + (left ? "left" : "right") + " hand, then press Record sign."
        : "Enable the camera to practice with your " + (left ? "left" : "right") + " hand.");
});

// // Decorative page background, independent from the webcam canvas and inference.
(() => {
    const surface = document.getElementById('ambientCanvas');
    const paint = surface.getContext('2d');
    const toggle = document.getElementById('motionToggle');
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    // Original geometric animal outlines, inspired by constellation illustrations.
    const shapes = {
        deer: [
            [[35,43],[30,60],[40,82],[50,92],[60,82],[70,60],[65,43],[50,39],[35,43]],
            [[35,50],[18,39],[22,51],[33,57]], [[65,50],[82,39],[78,51],[67,57]],
            [[39,45],[27,32],[20,17],[22,4]], [[27,32],[13,25],[8,12]], [[20,17],[11,7]], [[27,32],[32,18],[29,8]],
            [[61,45],[73,32],[80,17],[78,4]], [[73,32],[87,25],[92,12]], [[80,17],[89,7]], [[73,32],[68,18],[71,8]],
            [[37,61],[43,59],[45,65]], [[63,61],[57,59],[55,65]], [[40,82],[50,77],[60,82]], [[47,79],[50,84],[53,79]]
        ],
        owl: [
            [[18,13],[37,25],[63,25],[82,13],[79,44],[86,62],[75,83],[50,94],[25,83],[14,62],[21,44],[18,13]],
            [[21,44],[28,33],[42,34],[49,47],[41,60],[28,59],[21,44]],
            [[79,44],[72,33],[58,34],[51,47],[59,60],[72,59],[79,44]],
            [[33,43],[38,43],[38,48],[33,48],[33,43]], [[62,43],[67,43],[67,48],[62,48],[62,43]],
            [[45,58],[50,69],[55,58],[45,58]], [[21,55],[30,73],[50,86],[70,73],[79,55]],
            [[25,83],[36,67],[43,87]], [[75,83],[64,67],[57,87]]
        ],
        elephant: [
            [[37,26],[25,13],[10,18],[3,38],[10,58],[28,66],[37,51]],
            [[63,26],[75,13],[90,18],[97,38],[90,58],[72,66],[63,51]],
            [[37,19],[50,15],[63,19],[69,37],[62,58],[58,69],[58,88],[51,96],[44,91],[44,84],[49,87],[51,83],[48,63],[38,58],[31,37],[37,19]],
            [[38,43],[44,42]], [[56,42],[62,43]], [[38,55],[32,72],[44,63]], [[62,55],[68,72],[58,63]],
            [[10,18],[23,38],[10,58]], [[90,18],[77,38],[90,58]], [[23,38],[37,26]], [[77,38],[63,26]]
        ],
        whale: [
            [[4,42],[12,31],[30,29],[47,36],[65,45],[80,47],[89,34],[98,30],[94,44],[98,54],[87,56],[79,53],[65,65],[43,72],[23,66],[10,56],[4,42]],
            [[4,45],[23,49],[43,53],[65,53],[80,47]], [[10,56],[28,59],[47,62],[65,60]],
            [[40,58],[48,81],[60,87],[55,69]], [[18,39],[21,39]], [[29,30],[29,18],[23,13]], [[29,18],[35,11]]
        ],
        jellyfish: [
            [[18,42],[20,28],[31,15],[50,10],[69,15],[80,28],[82,42],[70,38],[60,45],[50,39],[40,45],[30,38],[18,42]],
            [[20,28],[40,25],[60,25],[80,28]], [[31,15],[40,25],[40,45]], [[69,15],[60,25],[60,45]],
            [[23,42],[26,57],[17,74],[20,95]], [[34,42],[38,57],[29,77],[34,91]],
            [[45,43],[43,61],[49,75],[42,98]], [[56,43],[61,59],[54,80],[58,96]],
            [[67,42],[72,60],[66,74],[73,91]], [[78,42],[83,56],[76,71],[82,86]]
        ],
        lion: [
            [[50,4],[65,10],[78,8],[83,23],[95,34],[90,50],[96,65],[81,76],[75,91],[59,89],[50,98],[40,89],[24,91],[19,76],[4,65],[10,50],[5,34],[17,23],[22,8],[35,10],[50,4]],
            [[30,28],[24,20],[22,33],[28,41]], [[70,28],[76,20],[78,33],[72,41]],
            [[30,28],[50,23],[70,28],[75,49],[66,71],[50,83],[34,71],[25,49],[30,28]],
            [[30,45],[40,42],[43,49]], [[70,45],[60,42],[57,49]],
            [[42,58],[50,55],[58,58],[50,66],[42,58]], [[50,66],[50,73],[39,69]], [[50,73],[61,69]],
            [[25,49],[34,71],[24,91]], [[75,49],[66,71],[75,91]], [[50,83],[50,98]]
        ]
    };
    const count = 160;
    // Sample each outline by length, retaining its own connections.
    const templates = Object.entries(shapes).map(([name, paths]) => {
        const segments = [];
        paths.forEach((path, pathId) => path.slice(1).forEach((b, i) => {
            const a = path[i]; segments.push({a,b,pathId,length:Math.hypot(b[0]-a[0],b[1]-a[1])});
        }));
        const total = segments.reduce((sum,s)=>sum+s.length,0), points=[], edges=[];
        let segment=0, offset=0;
        for(let i=0;i<count;i++) {
            const distance=i*total/count;
            while(segment<segments.length-1 && offset+segments[segment].length<distance) offset+=segments[segment++].length;
            const s=segments[segment], t=(distance-offset)/s.length;
            points.push({x:s.a[0]+(s.b[0]-s.a[0])*t,y:s.a[1]+(s.b[1]-s.a[1])*t,path:s.pathId});
            if(i && points[i-1].path===s.pathId) edges.push([i-1,i]);
        }
        // Sparse local cross-links give the outlines a constellation mesh.
        points.forEach((p,i)=>points.map((q,j)=>({j,d:Math.hypot(p.x-q.x,p.y-q.y)}))
            .filter(q=>q.j>i+2 && q.d>4 && q.d<17).sort((a,b)=>a.d-b.d).slice(0,2)
            .forEach(q=>edges.push([i,q.j])));
        return {name,points,edges};
    });
    let paused=false;
    try {paused=localStorage.getItem('signbridge-background-paused')==='true';} catch {}
    let width=0,height=0,raf=0,last=null,elapsedShape=0,hue=0,groups=[],particles=[];
    function pick(previous) {
        const choices=templates.filter(t=>t!==previous);
        return choices[Math.floor(Math.random()*choices.length)];
    }
    function updateToggle() {
        toggle.disabled=reduced.matches;
        toggle.textContent=reduced.matches?'Reduced motion':paused?'Resume background':'Pause background';
        toggle.setAttribute('aria-pressed',String(paused||reduced.matches));
    }
    function schedule(){cancelAnimationFrame(raf);last=null;raf=requestAnimationFrame(draw);}
    function resize(){
        width=innerWidth;height=innerHeight;
        const ratio=Math.min(devicePixelRatio||1,2);
        surface.width=Math.round(width*ratio);surface.height=Math.round(height*ratio);
        paint.setTransform(ratio,0,0,ratio,0,0);
        groups=Array.from({length:1},(_,i)=>{
            const target=pick(null);
            return {target,from:target.points.map(p=>({...p})),points:target.points.map(p=>({...p})),offset:i*150};
        });
        particles=Array.from({length:Math.min(85,Math.max(26,Math.round(width*height/19000)))},()=>{
            const angle=Math.random()*Math.PI*2, orbit=.18+Math.sqrt(Math.random())*.8;
            return {angle,orbit,speed:.035+Math.random()*.245,hue:Math.random()*360,
                r:.7+Math.random()*1.2,glow:Math.random()>.86};
        });
        elapsedShape=0;schedule();
    }
    function draw(now){
        if(document.hidden)return;
        if(last!==null && now-last<1000/30){raf=requestAnimationFrame(draw);return;}
        const dt=last===null?0:Math.min((now-last)/1000,.1);last=now;
        const still=paused||reduced.matches||document.body.classList.contains('recording');
        if(!still){
            hue=(hue+dt*24)%360;elapsedShape+=dt;
            if(elapsedShape>=3){
                elapsedShape%=3;
                groups.forEach(g=>{g.from=g.points.map(p=>({...p}));g.target=pick(g.target);});
            }
        }
        paint.clearRect(0,0,width,height);
        // Separate, thin orbital network behind the larger animal constellation.
        particles.forEach(p=>{
            if(!still)p.angle=(p.angle+p.speed*dt)%(Math.PI*2);
            p.x=width/2+Math.cos(p.angle)*width*.49*p.orbit;
            p.y=height/2+Math.sin(p.angle)*height*.49*p.orbit;
        });
        particles.forEach((p,i)=>{
            const color=(p.hue+hue)%360;
            for(let j=i+1;j<particles.length;j++){
                const q=particles[j],distance=Math.hypot(p.x-q.x,p.y-q.y);
                if(distance<180){
                    paint.strokeStyle=`hsla(${color},85%,72%,${.22*(1-distance/180)})`;
                    paint.lineWidth=.65;paint.beginPath();paint.moveTo(p.x,p.y);paint.lineTo(q.x,q.y);paint.stroke();
                }
            }
            const radius=p.glow?12:6;
            const glow=paint.createRadialGradient(p.x,p.y,0,p.x,p.y,radius);
            glow.addColorStop(0,`hsla(${color},90%,75%,.22)`);glow.addColorStop(1,`hsla(${color},90%,75%,0)`);
            paint.fillStyle=glow;paint.beginPath();paint.arc(p.x,p.y,radius,0,Math.PI*2);paint.fill();
            paint.fillStyle=`hsla(${color},90%,82%,.65)`;paint.beginPath();paint.arc(p.x,p.y,p.r,0,Math.PI*2);paint.fill();
        });
        const t=Math.min(elapsedShape/.95,1), blend=t*t*(3-2*t);
        groups.forEach((g,gi)=>{
            const size=Math.min(width*.9,height*.9,900);
            const cx=width*.5;
            const cy=height*.5;
            g.points=g.from.map((p,i)=>({x:p.x+(g.target.points[i].x-p.x)*blend,y:p.y+(g.target.points[i].y-p.y)*blend}));
            const screen=g.points.map(p=>({x:cx+(p.x-50)*size/100,y:cy+(p.y-50)*size/100}));
            g.target.edges.forEach(([a,b])=>{
                const p=screen[a],q=screen[b];
                paint.strokeStyle=`hsla(${(hue+g.offset+a*.7)%360},85%,72%,.32)`;
                paint.lineWidth=1.6;paint.beginPath();paint.moveTo(p.x,p.y);paint.lineTo(q.x,q.y);paint.stroke();
            });
            screen.forEach((p,i)=>{
                const color=(hue+g.offset+i*.7)%360;
                if(i%9===0){
                    const glow=paint.createRadialGradient(p.x,p.y,0,p.x,p.y,9);
                    glow.addColorStop(0,`hsla(${color},90%,70%,.25)`);glow.addColorStop(1,`hsla(${color},90%,70%,0)`);
                    paint.fillStyle=glow;paint.beginPath();paint.arc(p.x,p.y,9,0,Math.PI*2);paint.fill();
                }
                paint.fillStyle=`hsla(${color},90%,85%,.85)`;
                paint.beginPath();paint.arc(p.x,p.y,i%7===0?2:1.25,0,Math.PI*2);paint.fill();
            });
        });
        if(!paused&&!reduced.matches)raf=requestAnimationFrame(draw);
    }
    toggle.addEventListener('click',()=>{paused=!paused;try{localStorage.setItem('signbridge-background-paused',String(paused));}catch{}updateToggle();schedule();});
    window.addEventListener('resize',resize);
    document.addEventListener('visibilitychange',()=>{if(!document.hidden)schedule();else cancelAnimationFrame(raf);});
    reduced.addEventListener('change',()=>{updateToggle();schedule();});
    updateToggle();resize();
})();
