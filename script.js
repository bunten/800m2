
import {
    FilesetResolver,
    PoseLandmarker,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ========================================
// 1. HTMLの部品を取得
// ========================================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const aiStatus = document.getElementById("aiStatus");
const videoStatus = document.getElementById("videoStatus");

const sideInput = document.getElementById("sideVideo");
const frontInput = document.getElementById("frontVideo");

const sideButton = document.getElementById("sideButton");
const frontButton = document.getElementById("frontButton");

const trunkOutput = document.getElementById("trunk");
const shoulderOutput = document.getElementById("shoulder");
const hipOutput = document.getElementById("hip");
const kneeOutput = document.getElementById("knee");

const resultSide = document.getElementById("resultSide");
const resultFront = document.getElementById("resultFront");

const backFrameButton = document.getElementById("backFrame");
const forwardFrameButton = document.getElementById("forwardFrame");
const clearResultsButton = document.getElementById("clearResults");

// ========================================
// 2. アプリの状態
// ========================================
let videos = {
    side: null,
    front: null
};

let currentView = "side";
let poseLandmarker = null;
let lastDetectionTime = 0;

let metricsByView = {
    side: null,
    front: null
};

let animationStarted = false;

// ========================================
// 3. MediaPipeのAIモデルを読み込む
// ========================================
async function initializeAI() {
    try {
        aiStatus.textContent = "AIモデルを読み込んでいます...";

        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );

        poseLandmarker = await PoseLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath:
                        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                    delegate: "CPU"
                },
                runningMode: "VIDEO",
                numPoses: 1,
                minPoseDetectionConfidence: 0.5,
                minPosePresenceConfidence: 0.5,
                minTrackingConfidence: 0.5
            }
        );

        aiStatus.textContent =
            "AIモデルの準備ができました。動画を選択してください。";

        if (video.readyState >= 2) {
            analyzeCurrentFrame();
        }

        if (!animationStarted) {
            animationStarted = true;
            requestAnimationFrame(detectFrame);
        }

    } catch (error) {
        console.error("AIモデル読み込みエラー:", error);
        aiStatus.textContent =
            "AIモデルを読み込めませんでした。インターネット接続を確認してください。";
    }
}

// ========================================
// 4. 動画ファイルを読み込む
// ========================================
function loadVideo(file, view) {
    if (!file) return;

    if (videos[view]) {
        URL.revokeObjectURL(videos[view]);
    }

    videos[view] = URL.createObjectURL(file);
    metricsByView[view] = null;

    updateSavedResults();
    setView(view);
}

// ========================================
// 5. 横方向・正面方向を切り替える
// ========================================
function setView(view) {
    currentView = view;

    sideButton.classList.toggle("active", view === "side");
    frontButton.classList.toggle("active", view === "front");

    if (!videos[view]) {
        video.pause();
        video.removeAttribute("src");
        video.load();
        clearCanvasAndMetrics();

        videoStatus.textContent = "この方向の動画を選択してください。";
        return;
    }

    video.pause();
    video.src = videos[view];
    video.load();

    videoStatus.textContent =
        view === "side"
            ? "横方向の動画を表示しています。"
            : "正面方向の動画を表示しています。";

    clearCanvasAndMetrics();
}

sideInput.addEventListener("change", function () {
    loadVideo(this.files[0], "side");
});

frontInput.addEventListener("change", function () {
    loadVideo(this.files[0], "front");
});

sideButton.addEventListener("click", function () {
    setView("side");
});

frontButton.addEventListener("click", function () {
    setView("front");
});

// ========================================
// 6. 角度計算のための関数
// ========================================
function midpoint(a, b) {
    return {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2
    };
}

function horizontalAngle(a, b) {
    let angle =
        Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI;

    if (angle > 90) angle -= 180;
    if (angle < -90) angle += 180;

    return angle;
}

function trunkAngle(shoulder, hip) {
    return Math.atan2(
        shoulder.x - hip.x,
        hip.y - shoulder.y
    ) * 180 / Math.PI;
}

// ========================================
// 7. 骨格の位置から数値を計算する
// ========================================
function analyzePose(points) {
    // MediaPipeの番号
    // 11・12：左右の肩
    // 23・24：左右の腰
    // 25・26：左右の膝
    const LS = points[11];
    const RS = points[12];
    const LH = points[23];
    const RH = points[24];
    const LK = points[25];
    const RK = points[26];

    const required = [LS, RS, LH, RH];

    if (required.some(p => !p || (p.visibility ?? 1) < 0.5)) {
        aiStatus.textContent =
            "肩または腰を十分に検出できません。全身が映るように撮影してください。";
        return;
    }

    const shoulderCenter = midpoint(LS, RS);
    const hipCenter = midpoint(LH, RH);

    const trunk = Math.abs(
        trunkAngle(shoulderCenter, hipCenter)
    );

    const shoulder = Math.abs(
        horizontalAngle(LS, RS)
    );

    const hip = Math.abs(
        horizontalAngle(LH, RH)
    );

    let knee = null;

    if (
        LK && RK &&
        (LK.visibility ?? 1) >= 0.5 &&
        (RK.visibility ?? 1) >= 0.5
    ) {
        knee = Math.abs(LK.y - RK.y) * video.videoHeight;
    }

    const result = {
        direction: currentView,
        trunkAngle: trunk.toFixed(1),
        shoulderAngle: shoulder.toFixed(1),
        hipAngle: hip.toFixed(1),
        kneeDifference:
            knee === null ? "検出できません" : knee.toFixed(1) + "px"
    };

    metricsByView[currentView] = result;

    displayMetrics(result);
    updateSavedResults();

    aiStatus.textContent =
        "骨格を検出しています。動画を再生すると数値が更新されます。";
}

// ========================================
// 8. 数値を画面に表示する
// ========================================
function displayMetrics(m) {
    trunkOutput.textContent = m.trunkAngle + "°";
    shoulderOutput.textContent = m.shoulderAngle + "°";
    hipOutput.textContent = m.hipAngle + "°";
    kneeOutput.textContent = m.kneeDifference;
}

// ========================================
// 9. 骨格を動画の上に描く
// ========================================
function drawPose(points) {
    const drawing = new DrawingUtils(ctx);

    drawing.drawConnectors(
        points,
        PoseLandmarker.POSE_CONNECTIONS,
        {
            color: "#00ff88",
            lineWidth: 4
        }
    );

    drawing.drawLandmarks(
        points,
        {
            color: "#ff4444",
            lineWidth: 2,
            radius: 4
        }
    );
}

// ========================================
// 10. 動画の現在の1コマを分析する
// ========================================
function analyzeCurrentFrame() {
    if (!poseLandmarker) return;
    if (video.readyState < 2 || !video.videoWidth) return;

    if (
        canvas.width !== video.videoWidth ||
        canvas.height !== video.videoHeight
    ) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
        const result = poseLandmarker.detectForVideo(
            video,
            performance.now()
        );

        if (result.landmarks && result.landmarks.length > 0) {
            const points = result.landmarks[0];
            drawPose(points);
            analyzePose(points);
        } else {
            aiStatus.textContent =
                "人物を検出できません。全身が映る位置から撮影してください。";
        }
    } catch (error) {
        console.error("動画分析エラー:", error);
        aiStatus.textContent =
            "動画の分析中にエラーが発生しました。ブラウザのConsoleを確認してください。";
    }
}

// ========================================
// 11. 動画再生中、約0.12秒ごとに分析する
// ========================================
function detectFrame() {
    if (
        poseLandmarker &&
        !video.paused &&
        !video.ended &&
        video.readyState >= 2
    ) {
        const now = performance.now();

        if (now - lastDetectionTime >= 120) {
            lastDetectionTime = now;
            analyzeCurrentFrame();
        }
    }

    requestAnimationFrame(detectFrame);
}

// ========================================
// 12. 動画の読み込み・再生・停止に対応
// ========================================
video.addEventListener("loadeddata", function () {
    if (poseLandmarker) analyzeCurrentFrame();
});

video.addEventListener("seeked", function () {
    if (poseLandmarker) analyzeCurrentFrame();
});

video.addEventListener("play", function () {
    if (poseLandmarker) analyzeCurrentFrame();
});

video.addEventListener("timeupdate", function () {
    if (poseLandmarker && video.paused) {
        analyzeCurrentFrame();
    }
});

// ========================================
// 13. 分析結果を⑤に保存表示する
// ========================================
function updateSavedResults() {
    const side = metricsByView.side;
    const front = metricsByView.front;

    resultSide.textContent = side
        ? `横方向：体幹 ${side.trunkAngle}°、肩 ${side.shoulderAngle}°、腰 ${side.hipAngle}°、膝の高さの差 ${side.kneeDifference}`
        : "横方向：まだ分析していません。";

    resultFront.textContent = front
        ? `正面方向：体幹 ${front.trunkAngle}°、肩 ${front.shoulderAngle}°、腰 ${front.hipAngle}°、膝の高さの差 ${front.kneeDifference}`
        : "正面方向：まだ分析していません。";
}

// ========================================
// 14. 数値と骨格表示をリセットする
// ========================================
function clearCanvasAndMetrics() {
    trunkOutput.textContent = "--";
    shoulderOutput.textContent = "--";
    hipOutput.textContent = "--";
    kneeOutput.textContent = "--";

    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ========================================
// 15. 1コマ戻る・1コマ進む
// ========================================
backFrameButton.addEventListener("click", function () {
    if (!video.src || !video.duration) return;

    video.pause();
    video.currentTime = Math.max(0, video.currentTime - 1 / 30);
});

forwardFrameButton.addEventListener("click", function () {
    if (!video.src || !video.duration) return;

    video.pause();
    video.currentTime = Math.min(
        video.duration,
        video.currentTime + 1 / 30
    );
});

// ========================================
// 16. 分析結果をリセットする
// ========================================
clearResultsButton.addEventListener("click", function () {
    metricsByView.side = null;
    metricsByView.front = null;

    updateSavedResults();
    clearCanvasAndMetrics();

    aiStatus.textContent =
        "分析結果をリセットしました。動画を再生して再分析できます。";
});

// ========================================
// 17. アプリを起動する
// ========================================
initializeAI();
