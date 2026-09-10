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
    document.getElementById("practiceTitle").textContent = "Alphabet Practice";
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
const API = ["5500", "5501"].includes(location.port) ? "http://127.0.0.1:5010" : "";
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

function updateFeedback(result) {
    const percentage = Math.round(result.confidence * 100);
    document.getElementById("confidence").textContent = percentage + "%";
    document.getElementById("progressBar").style.width = percentage + "%";
    if (result.confidence < 0.7) {
        lastCandidate = "";
        repeatCount = 0;
        document.getElementById("gesture").textContent = "Uncertain";
        showFeedback("Try better lighting and keep your whole hand in view.");
        return;
    }
    repeatCount = result.letter === lastCandidate ? repeatCount + 1 : 1;
    lastCandidate = result.letter;
    document.getElementById("gesture").textContent = repeatCount >= 3 ? result.letter : "Checking…";
    showFeedback(repeatCount >= 3
        ? (["J", "Z"].includes(result.letter)
            ? "Image resembles " + result.letter + ". Its signing motion is not checked yet."
            : "Detected: " + result.letter)
        : "Hold your sign steady…");
}

// Submit raw camera pixels. The server performs the training-time crop.
function captureHand() {
    if (!video.videoWidth || !video.videoHeight) return null;
    const frame = document.createElement("canvas");
    const scale = Math.min(1, 960 / Math.max(video.videoWidth, video.videoHeight));
    frame.width = Math.round(video.videoWidth * scale);
    frame.height = Math.round(video.videoHeight * scale);
    frame.getContext("2d").drawImage(video, 0, 0, frame.width, frame.height);
    return frame.toDataURL("image/jpeg", 0.95);
}

async function predictFrame(current, landmarks) {
    const image = captureHand(landmarks);
    if (!image) return;
    current.busy = true;
    current.lastRequest = performance.now();
    const handVersion = current.handVersion;
    const controller = new AbortController();
    current.predictionController = controller;
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
        const response = await fetch(API + "/predict", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({image}), signal: controller.signal
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Prediction failed");
        if (session !== current || handVersion !== current.handVersion || !current.hasHand) return;
        if (result.status === "no_hand") {
            resetPrediction("No clear hand");
            document.getElementById("modelInput").removeAttribute("src");
            showFeedback(result.message);
            return;
        }
        if (result.crop?.startsWith("data:image/png;base64,")) document.getElementById("modelInput").src = result.crop;
        if (!/^[A-Z]$/.test(result.letter) || !Number.isFinite(result.confidence)
            || result.confidence < 0 || result.confidence > 1) throw new Error("Invalid model response");
        if (session !== current || handVersion !== current.handVersion || !current.hasHand) return;
        document.getElementById("modelStatus").textContent = "ASL-HG model connected";
        updateFeedback(result);
    } catch (error) {
        if (session !== current || handVersion !== current.handVersion) return;
        console.error(error);
        resetPrediction();
        document.getElementById("modelStatus").textContent = "Connection interrupted";
        showFeedback("Cannot get a prediction. Keep the SignBridge server running; retrying…");
        current.lastRequest = performance.now() + 1500;
    } finally {
        clearTimeout(timeout);
        current.busy = false;
        if (current.predictionController === controller) current.predictionController = null;
    }
}

function onResults(current, results) {
    if (session !== current) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const landmarks = results.multiHandLandmarks?.[0];
    if (!landmarks) {
        if (current.hasHand) {
            current.handVersion++;
            current.predictionController?.abort();
        }
        current.hasHand = false;
        resetPrediction("No hand");
        document.getElementById("modelInput").removeAttribute("src");
        showFeedback("Show one hand clearly to the camera.");
        return;
    }
    current.hasHand = true;
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {color: "#8b5cf6", lineWidth: 4});
    drawLandmarks(ctx, landmarks, {color: "#ffffff", lineWidth: 2});
    if (!current.busy && performance.now() - current.lastRequest >= 500) {
        void predictFrame(current, landmarks);
    }
}

async function startCamera() {
    if (session) return;
    const current = {busy: false, hasHand: false, handVersion: 0, lastRequest: -Infinity,
        stream: null, hands: null, raf: null, sending: null, controller: new AbortController()};
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
        const health = await response.json();
        if (!response.ok || health.status !== "ready" || health.model !== "aslhg_candidate.keras"
            || health.preprocessing !== "raw-hand-v1") throw new Error("Start the new ASL-HG server and open http://127.0.0.1:5010.");
        if (session !== current) return;
        if (typeof Hands === "undefined" || typeof drawLandmarks === "undefined") {
            throw new Error("Hand tracking could not load. Check your internet connection and refresh.");
        }
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("Open SignBridge at http://127.0.0.1:5010 to use the camera.");
        showFeedback("Allow camera access to start practicing.");
        const stream = await navigator.mediaDevices.getUserMedia({video: {width: 960, height: 720}, audio: false});
        if (session !== current) { stream.getTracks().forEach(track => track.stop()); return; }
        current.stream = stream;
        video.srcObject = stream;
        await video.play();
        if (session !== current) return;
        current.hands = new Hands({locateFile: file => "https://cdn.jsdelivr.net/npm/@mediapipe/hands/" + file});
        current.hands.setOptions({maxNumHands: 1, modelComplexity: 1,
            minDetectionConfidence: 0.6, minTrackingConfidence: 0.6});
        current.hands.onResults(results => onResults(current, results));
        document.getElementById("cameraPlaceholder").style.display = "none";
        document.getElementById("modelStatus").textContent = "ASL-HG model connected";
        start.textContent = "Camera running";
        showFeedback("Loading hand tracking. Keep your hand in view…");
        const frame = async () => {
            if (session !== current) return;
            try {
                current.sending = current.hands.send({image: video});
                await current.sending;
                if (session === current) current.raf = requestAnimationFrame(frame);
            } catch (error) {
                if (session === current) {
                    stopCamera();
                    showFeedback("Hand tracking stopped. Check your connection, then enable the camera again.");
                    console.error(error);
                }
            }
        };
        current.raf = requestAnimationFrame(frame);
    } catch (error) {
        if (session !== current) return;
        stopCamera();
        document.getElementById("modelStatus").textContent = "Not connected";
        const cameraErrors = {NotAllowedError: "Camera permission was denied. Allow it in your browser, then try again.",
            NotFoundError: "No camera was found. Connect a webcam and try again.",
            NotReadableError: "The camera is busy. Close other camera apps and try again."};
        showFeedback(cameraErrors[error.name] || (error.name === "TypeError" || error.name === "AbortError"
            ? "Start the SignBridge server, open http://127.0.0.1:5010, and try again." : error.message));
        console.error(error);
    }
}

function stopCamera() {
    const previous = session;
    session = null;
    if (previous) {
        previous.controller.abort();
        previous.predictionController?.abort();
        cancelAnimationFrame(previous.raf);
        previous.stream?.getTracks().forEach(track => track.stop());
        // Close MediaPipe only after an in-flight frame has finished.
        Promise.resolve(previous.sending).catch(() => {}).then(() => previous.hands?.close()).catch(console.error);
    }
    video.srcObject = null;
    document.getElementById("modelInput").removeAttribute("src");
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

