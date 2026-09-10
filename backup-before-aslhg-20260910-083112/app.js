// =============================
// PAGE NAVIGATION
// =============================

function showPage(page) {
    if (page !== "learn") stopCamera();

    const pages =
        document.querySelectorAll(".page");

    pages.forEach(p => {

        p.classList.remove("active");

    });


    document
        .getElementById(page)
        .classList.add("active");

}


// =============================
// LEARNING PRACTICE
// =============================

function openPractice(category = "Practice") {

    showPage("learn");

    const practice =
        document.getElementById("practice");

    practice.classList.remove("hidden");


    document
        .getElementById("practiceTitle")
        .innerText =
        category + " Practice";


    practice.scrollIntoView({

        behavior: "smooth"

    });

}


function closePractice() {

    stopCamera();

    document
        .getElementById("practice")
        .classList.add("hidden");

}


// =============================
// CAMERA + MEDIAPIPE
// =============================

const video =
    document.getElementById("video");

const canvas =
    document.getElementById("canvas");

const ctx =
    canvas.getContext("2d");


let camera = null;
let hands = null;


// =============================
// MEDIAPIPE RESULT
// =============================

function onResults(results) {

    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;


    ctx.clearRect(

        0,
        0,
        canvas.width,
        canvas.height

    );


    if (
        results.multiHandLandmarks &&
        results.multiHandLandmarks.length > 0
    ) {

        const landmarks =
            results.multiHandLandmarks[0];


        // Draw hand skeleton

        drawConnectors(

            ctx,

            landmarks,

            HAND_CONNECTIONS,

            {
                color:
                    "#8b5cf6",

                lineWidth:
                    4
            }

        );


        drawLandmarks(

            ctx,

            landmarks,

            {
                color:
                    "#ffffff",

                lineWidth:
                    2
            }

        );


        // Tracking only: no classifier or fixed confidence values.
        document.getElementById("gesture").innerText = "Hand detected";
        document.getElementById("confidence").innerText = "21 landmarks";
        document.getElementById("progressBar").style.width = "0%";
        document.getElementById("feedback").innerText = "Hand tracking is working. Move your hand to see the landmarks follow.";

    }

    else {

        document
            .getElementById("gesture")
            .innerText =
            "No hand";
        document.getElementById("feedback").innerText = "Show one hand to the camera.";


        document
            .getElementById("confidence")
            .innerText =
            "—";


        document
            .getElementById("progressBar")
            .style.width =
            "0%";

    }

}



// =============================
// START CAMERA
// =============================

let cameraStarting = false;
async function startCamera() {
    if (cameraStarting || (video.srcObject && video.srcObject.active)) return;
    cameraStarting = true;

    try {

        hands =
            new Hands({

                locateFile:
                    (file) => {

                        return (
                            "https://cdn.jsdelivr.net/npm/@mediapipe/hands/" +
                            file
                        );

                    }

            });


        hands.setOptions({

            maxNumHands:
                1,

            modelComplexity:
                1,

            minDetectionConfidence:
                0.6,

            minTrackingConfidence:
                0.6

        });


        hands.onResults(
            onResults
        );


        camera =
            new Camera(

                video,

                {

                    onFrame:
                        async () => {

                            await hands.send({

                                image:
                                    video

                            });

                        },

                    width:
                        960,

                    height:
                        720

                }

            );


        await camera.start();
        cameraStarting = false;


        document
            .getElementById(
                "cameraPlaceholder"
            )
            .style.display =
            "none";

    }

    catch (error) {
        cameraStarting = false;
        stopCamera();

        console.error(
            error
        );


        alert(

            "Camera could not start. " +
            "Please allow camera permission."

        );

    }

}



// =============================
// STOP CAMERA
// =============================

function stopCamera() {
    document.getElementById("gesture").innerText = "—";
    document.getElementById("confidence").innerText = "—";
    document.getElementById("feedback").innerText = "Enable the camera to track your hand.";
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (camera) {

        camera.stop();

    }


    if (video.srcObject) {

        const tracks =
            video
                .srcObject
                .getTracks();


        tracks.forEach(

            track =>
                track.stop()

        );

    }


    document
        .getElementById(
            "cameraPlaceholder"
        )
        .style.display =
        "flex";

}



// =============================
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
window.addEventListener("pagehide", stopCamera);
