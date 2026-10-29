// Navigation and practice
function showPage(page) {
    if (page !== "learn") stopCamera();
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    document.getElementById(page).classList.add("active");
}

function openPractice(category = "Alphabet") {
    showPage("learn");
    const practice = document.getElementById("practice");
    practice.classList.remove("hidden");
    document.getElementById("practiceTitle").textContent = isNumberReference() ? "Number Reference" : "Alphabet Practice";
    if (!["Alphabet", "Practice"].includes(category)) {
        showFeedback("Alphabet recognition is available now. Vocabulary and sentences are coming later.");
    }
    practice.scrollIntoView({behavior: "smooth"});
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
// Local previews can use any port; only Flask on 5010 serves the model API.
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
        const response = await fetch(API + "/predict-sequence", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({frames, timestamps, hand: side === "left" ? "Left" : "Right"}),
            signal: controller.signal
        });
        const result = await readModelJSON(response);
        if (!valid()) return;
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


// SIGN LIBRARY
// =============================

const signs = [

    {
        name:
            "Hello",

        emoji:
            "👋",

        category:
            "Vocabulary"
    },

    {
        name:
            "Thank You",

        emoji:
            "🙏",

        category:
            "Vocabulary"
    },

    {
        name:
            "A",

        emoji:
            "✊",

        category:
            "Alphabet"
    },

    {
        name:
            "B",

        emoji:
            "🖐️",

        category:
            "Alphabet"
    },

    {
        name:
            "I Love You",

        emoji:
            "🤟",

        category:
            "Sentence"
    },

    {
        name:
            "Yes",

        emoji:
            "👍",

        category:
            "Vocabulary"
    },

    {
        name:
            "No",

        emoji:
            "✋",

        category:
            "Vocabulary"
    },

    {
        name:
            "Help",

        emoji:
            "🫶",

        category:
            "Vocabulary"
    }

];



function displaySigns(
    signArray
) {

    const list =
        document.getElementById(
            "signList"
        );


    list.innerHTML =
        "";


    signArray.forEach(
        sign => {

            list.innerHTML += `

            <div class="sign-item">

                <div class="sign-emoji">

                    ${sign.emoji}

                </div>

                <h3>

                    ${sign.name}

                </h3>

                <p>

                    ${sign.category}

                </p>

            </div>

            `;

        }
    );

}



function searchSigns() {

    const text =
        document
            .getElementById(
                "search"
            )
            .value
            .toLowerCase();


    const result =
        signs.filter(
            sign =>

                sign.name
                    .toLowerCase()
                    .includes(text)

                ||

                sign.category
                    .toLowerCase()
                    .includes(text)

        );


    displaySigns(
        result
    );

}


displaySigns(
    signs
);



// =============================
// GAME
// =============================

let gameTimer;


function startGame() {

    clearInterval(
        gameTimer
    );


    let time =
        60;


    let score =
        0;


    document
        .getElementById("time")
        .innerText =
        time;


    document
        .getElementById("score")
        .innerText =
        score;


    gameTimer =
        setInterval(
            () => {

                time--;


                document
                    .getElementById(
                        "time"
                    )
                    .innerText =
                    time;


                if (
                    time <= 0
                ) {

                    clearInterval(
                        gameTimer
                    );


                    alert(
                        "Time's up!"
                    );

                }

            },

            1000

        );

}


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
    recordStatus.textContent = "Press Record sign when ready.";
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
    const left = document.getElementById("practiceHand").value === "left";
    document.getElementById("handModeNote").textContent = left
        ? "Show your left hand. Recognition pauses if the other hand is detected."
        : "Show your right hand. Recognition pauses if the other hand is detected.";
    if (isNumberReference()) showNumberReference();
    else showFeedback(session ? "Show your " + (left ? "left" : "right") + " hand, then press Record sign."
        : "Enable the camera to practice with your " + (left ? "left" : "right") + " hand.");
});
