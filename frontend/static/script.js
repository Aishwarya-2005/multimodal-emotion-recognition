document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - File Upload Form
    const fileUploadForm = document.getElementById("fileUploadForm");
    const faceInput = document.getElementById("faceInput");
    const audioInput = document.getElementById("audioInput");
    const faceDropZone = document.getElementById("faceDropZone");
    const audioDropZone = document.getElementById("audioDropZone");
    const faceFileName = document.getElementById("faceFileName");
    const audioFileName = document.getElementById("audioFileName");
    const clearFaceBtn = document.getElementById("clearFace");
    const clearAudioBtn = document.getElementById("clearAudio");

    // DOM Elements - Live Telemetry
    const startCameraBtn = document.getElementById("startCamera");
    const captureBtn = document.getElementById("capture");
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const scannerLine = document.getElementById("scannerLine");
    const recDot = document.getElementById("recDot");
    const telemetryStatusBadge = document.getElementById("telemetryStatusBadge");
    const statusEl = document.getElementById("status");

    // DOM Elements - Results Box
    const resultBox = document.getElementById("resultBox");
    const faceOut = document.getElementById("faceOut");
    const audioOut = document.getElementById("audioOut");
    const finalOut = document.getElementById("finalOut");
    const faceConfidence = document.getElementById("faceConfidence");
    const audioConfidence = document.getElementById("audioConfidence");
    const faceResultTile = document.getElementById("faceResultTile");
    const audioResultTile = document.getElementById("audioResultTile");
    const finalResultTile = document.getElementById("finalResultTile");

    // DOM Elements - Toast & Loading Overlay
    const loadingOverlay = document.getElementById("loadingOverlay");
    const loaderSubtext = document.getElementById("loaderSubtext");
    const validationToast = document.getElementById("validationToast");
    const toastMessage = document.getElementById("toastMessage");

    // Telemetry State Variables
    let mediaStream = null;
    let audioContext = null;
    let audioSource = null;
    let processor = null;
    let audioBuffer = [];
    let isRecording = false;

    // Toast Timer Reference
    let toastTimeout = null;

    // ==========================================
    // 1. Toast Alert & Loader Helper Functions
    // ==========================================
    function showToast(message) {
        if (toastTimeout) {
            clearTimeout(toastTimeout);
            validationToast.classList.remove("active");
            // Force reflow
            void validationToast.offsetWidth;
        }

        toastMessage.textContent = message;
        validationToast.style.display = "block";
        validationToast.classList.add("active");

        toastTimeout = setTimeout(() => {
            validationToast.classList.remove("active");
            setTimeout(() => {
                validationToast.style.display = "none";
            }, 300); // matches slide-out transition time
        }, 4000);
    }

    function showLoader(subtext) {
        loaderSubtext.textContent = subtext || "Executing neural network logic...";
        loadingOverlay.style.display = "flex";
        loadingOverlay.classList.add("active");
    }

    function hideLoader() {
        loadingOverlay.classList.remove("active");
        setTimeout(() => {
            loadingOverlay.style.display = "none";
        }, 300);
    }

    // ==========================================
    // 2. Drag and Drop File Interactions
    // ==========================================
    const registerDragEvents = (dropZone, fileInput, fileNameEl, clearBtn) => {
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add("highlight");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove("highlight");
            }, false);
        });

        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                updateFileDisplay(fileInput, fileNameEl, clearBtn);
            }
        });
    };

    function updateFileDisplay(input, labelEl, clearBtn) {
        if (input.files && input.files.length > 0) {
            labelEl.textContent = input.files[0].name;
            labelEl.classList.add("file-selected");
            clearBtn.style.display = "inline-block";
        } else {
            labelEl.textContent = "No file chosen";
            labelEl.classList.remove("file-selected");
            clearBtn.style.display = "none";
        }
    }

    // Register Drag/Drop Events
    registerDragEvents(faceDropZone, faceInput, faceFileName, clearFaceBtn);
    registerDragEvents(audioDropZone, audioInput, audioFileName, clearAudioBtn);

    // File Input Click Change Events
    faceInput.addEventListener("change", () => updateFileDisplay(faceInput, faceFileName, clearFaceBtn));
    audioInput.addEventListener("change", () => updateFileDisplay(audioInput, audioFileName, clearAudioBtn));

    // Clear Button Clicks
    clearFaceBtn.addEventListener("click", (e) => {
        e.preventDefault();
        faceInput.value = "";
        updateFileDisplay(faceInput, faceFileName, clearFaceBtn);
    });

    clearAudioBtn.addEventListener("click", (e) => {
        e.preventDefault();
        audioInput.value = "";
        updateFileDisplay(audioInput, audioFileName, clearAudioBtn);
    });

    // ==========================================
    // 3. Update Result Presentation UI
    // ==========================================
    function displayResults(data) {
        if (!data || !data.success) {
            showToast(data.error || "A processing error occurred.");
            return;
        }

        // Display results block with opacity transition
        resultBox.style.display = "block";
        resultBox.scrollIntoView({ behavior: "smooth" });

        // Helper to update text content and glowing color badge
        const updateTile = (tileEl, outputEl, emotion) => {
            outputEl.textContent = emotion.toUpperCase();
            
            // Clean previous emotion classes
            tileEl.className = tileEl.classList.contains("final-tile") 
                ? "result-tile final-tile" 
                : "result-tile";
                
            // Apply corresponding neon accent class
            const formattedEmotion = emotion.trim().toLowerCase();
            tileEl.classList.add(`emo-${formattedEmotion}`);
        };

        updateTile(faceResultTile, faceOut, data.face);
	updateTile(audioResultTile, audioOut, data.audio);
	updateTile(finalResultTile, finalOut, data.result);	

	faceConfidence.textContent =
    	`Confidence: ${Number(data.face_confidence).toFixed(2)}%`;

	audioConfidence.textContent =
    	`Confidence: ${Number(data.audio_confidence).toFixed(2)}%`;
    	}

    // ==========================================
    // 4. File Upload Form submission (AJAX)
    // ==========================================
    fileUploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const imageFile = faceInput.files[0];
        const audioFile = audioInput.files[0];

        // Custom validation check
        if (!imageFile || !audioFile) {
            let missingMsg = "Please upload both files first.";
            if (!imageFile && audioFile) missingMsg = "Missing required Face Image upload.";
            if (imageFile && !audioFile) missingMsg = "Missing required Audio File upload.";
            showToast(missingMsg);
            return;
        }

        showLoader("Uploading audio and image files to server...");

        const formData = new FormData();
        formData.append("image", imageFile);
        formData.append("audio", audioFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Server returned error status ${response.status}`);
            }

            displayResults(data);
        } catch (err) {
            console.error(err);
            showToast(err.message || "Failed to execute prediction. See console for details.");
        } finally {
            hideLoader();
        }
    });

    // ==========================================
    // 5. Live Capture Telemetry logic
    // ==========================================
    async function startCameraAndMic() {
        statusEl.textContent = "Initializing camera & mic sensor arrays...";
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: true
            });
            video.srcObject = mediaStream;
            
            // Activate live scanner UI
            scannerLine.style.display = "block";
            telemetryStatusBadge.textContent = "ACTIVE";
            telemetryStatusBadge.classList.add("active-pulse");
            
            captureBtn.disabled = false;
            statusEl.textContent = "Sensor telemetry online. System calibrated.";
            startCameraBtn.disabled = true;
            startCameraBtn.classList.add("disabled");
        } catch (err) {
            console.error(err);
            statusEl.textContent = "Sensor initialization failed. Grant permissions.";
            showToast("Camera/Microphone access denied or unavailable.");
        }
    }

    function encodeWavFromFloat32(float32Array, sampleRate) {
        const numChannels = 1;
        const bytesPerSample = 2;
        const blockAlign = numChannels * bytesPerSample;
        const byteRate = sampleRate * blockAlign;
        const dataSize = float32Array.length * bytesPerSample;
        
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        function writeString(offset, str) {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        }

        // WAV Header Formats
        writeString(0, "RIFF");
        view.setUint32(4, 36 + dataSize, true);
        writeString(8, "WAVE");
        writeString(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM Format
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, 16, true); // Bits per sample
        writeString(36, "data");
        view.setUint32(40, dataSize, true);

        // Map float32 [-1.0, 1.0] samples to 16-bit signed PCM
        let offset = 44;
        for (let i = 0; i < float32Array.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }

        return new Blob([buffer], { type: "audio/wav" });
    }

    async function captureTelemetry() {
        if (!mediaStream) {
            showToast("Telemetry camera sensors are uninitialized.");
            return;
        }

        if (isRecording) return;
        isRecording = true;
        
        // Show recording telemetry UI
        captureBtn.disabled = true;
        recDot.style.display = "block";
        statusEl.textContent = "RECORDING ACOUSTIC ENERGY STREAM...";
        statusEl.classList.add("recording-pulse");

        audioBuffer = [];
        audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
        
        if (audioContext.state === "suspended") {
            await audioContext.resume();
        }

        const sampleRate = audioContext.sampleRate;

        // Disconnect old nodes if they exist
        if (audioSource) {
            try { audioSource.disconnect(); } catch (e) {}
        }
        if (processor) {
            try { processor.disconnect(); } catch (e) {}
        }

        // Attach nodes to capture mic
        audioSource = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (e) => {
            const input = e.inputBuffer.getChannelData(0);
            audioBuffer.push(new Float32Array(input));
        };

        audioSource.connect(processor);
        processor.connect(audioContext.destination);

        // Immediate snapshot of the face frame
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, width, height);

        // Recording interval timer updates (3 seconds)
        let secondsLeft = 3;
        const intervalId = setInterval(() => {
            secondsLeft -= 1;
            if (secondsLeft > 0) {
                statusEl.textContent = `CAPTURING ACOUSTIC ENERGY STREAM (${secondsLeft}s left)...`;
            } else {
                clearInterval(intervalId);
            }
        }, 1000);

        // Stop recording after 3 seconds and submit blobs
        setTimeout(async () => {
            // Teardown audio nodes
            try {
                audioSource.disconnect();
                processor.disconnect();
            } catch (e) {}

            recDot.style.display = "none";
            statusEl.classList.remove("recording-pulse");
            statusEl.textContent = "Encoding recorded WAV stream and extracting frame...";

            // Flatten float32 sub-chunks into single array
            const totalLength = audioBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
            const flatBuffer = new Float32Array(totalLength);
            let offset = 0;
            for (const chunk of audioBuffer) {
                flatBuffer.set(chunk, offset);
                offset += chunk.length;
            }

            const audioBlob = encodeWavFromFloat32(flatBuffer, sampleRate);

            canvas.toBlob(async (imageBlob) => {
                showLoader("Sending telemetry streams to classifier models...");

                const formData = new FormData();
                formData.append("image", imageBlob, "webcam.png");
                formData.append("audio", audioBlob, "mic.wav");

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.error || `Server error ${response.status}`);
                    }

                    displayResults(data);
                    statusEl.textContent = "Telemetry analysis completed. Models executed.";
                } catch (err) {
                    console.error(err);
                    showToast(err.message || "Failed to process telemetry datastream.");
                    statusEl.textContent = "Sensor telemetry prediction failed.";
                } finally {
                    hideLoader();
                    captureBtn.disabled = false;
                    isRecording = false;
                }
            }, "image/png");
        }, 3000);
    }

    startCameraBtn.addEventListener("click", startCameraAndMic);
    captureBtn.addEventListener("click", captureTelemetry);
});
