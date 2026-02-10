/**
 * S3 Upload Module for Game Study
 * Handles uploading session data to AWS S3 via Lambda
 *
 * Usage:
 * 1. Include this script: <script src="s3-uploader.js"></script>
 * 2. Set the API URL: S3Uploader.setApiUrl('https://your-api-gateway-url.amazonaws.com/prod')
 * 3. Call S3Uploader.upload(...) when game completes
 */

const S3Uploader = (function() {
    // Configuration
    const config = {
        apiBaseUrl: '', // Set via S3Uploader.setApiUrl()
        maxRetries: 3,
        retryDelayMs: 1000
    };

    // Upload state
    let uploadInProgress = false;
    let uploadStatus = 'idle'; // 'idle', 'uploading', 'success', 'error'

    /**
     * Initialize upload session - gets presigned URLs from Lambda
     */
    async function initUpload(sessionCode, gameName) {
        const response = await fetch(`${config.apiBaseUrl}/init-upload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionCode, gameName })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Init upload failed: ${response.status} - ${errorText}`);
        }

        return response.json();
    }

    /**
     * Upload audio directly to S3 via presigned URL
     */
    async function uploadAudio(presignedUrl, audioBlob) {
        const response = await fetch(presignedUrl, {
            method: 'PUT',
            headers: { 'Content-Type': 'audio/webm' },
            body: audioBlob
        });

        if (!response.ok) {
            throw new Error(`Audio upload failed: ${response.status}`);
        }

        return true;
    }

    /**
     * Submit session data to Lambda
     */
    async function submitSessionData(basePath, sessionData, eyeTrackingData, audioKey) {
        const response = await fetch(`${config.apiBaseUrl}/submit-session`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                basePath,
                sessionData,
                eyeTrackingData,
                audioKey
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Session submit failed: ${response.status} - ${errorText}`);
        }

        return response.json();
    }

    /**
     * Main upload function - orchestrates the full upload flow
     * @param {string} sessionCode - The session code (e.g., "1234")
     * @param {string} gameName - The game name (e.g., "Game A")
     * @param {object} sessionData - The session data object with movements, etc.
     * @param {object} eyeTrackingData - Eye tracking data with gaze array
     * @param {Array} audioChunks - Array of audio Blob chunks from MediaRecorder
     * @param {function} onProgress - Optional callback for progress updates
     */
    async function uploadSession(sessionCode, gameName, sessionData, eyeTrackingData, audioChunks, onProgress) {
        if (uploadInProgress) {
            console.warn('S3Uploader: Upload already in progress');
            return { success: false, error: 'Upload already in progress' };
        }

        if (!config.apiBaseUrl) {
            console.error('S3Uploader: API URL not configured. Call S3Uploader.setApiUrl() first.');
            return { success: false, error: 'API URL not configured' };
        }

        uploadInProgress = true;
        uploadStatus = 'uploading';

        try {
            // Step 1: Initialize upload session
            onProgress && onProgress('Initializing upload...');
            const initResult = await initUpload(sessionCode, gameName);
            const { basePath, audioUploadUrl, audioKey } = initResult;

            // Step 2: Upload audio if available
            if (audioChunks && audioChunks.length > 0) {
                onProgress && onProgress('Uploading audio...');
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                await uploadAudio(audioUploadUrl, audioBlob);
            }

            // Step 3: Submit session data
            onProgress && onProgress('Uploading session data...');
            const result = await submitSessionData(basePath, sessionData, eyeTrackingData, audioKey);

            uploadStatus = 'success';
            onProgress && onProgress('Upload complete!');

            return { success: true, ...result };

        } catch (error) {
            console.error('S3Uploader: Upload failed:', error);
            uploadStatus = 'error';
            onProgress && onProgress(`Upload failed: ${error.message}`);

            return { success: false, error: error.message };

        } finally {
            uploadInProgress = false;
        }
    }

    /**
     * Upload with retry logic and exponential backoff
     */
    async function uploadWithRetry(sessionCode, gameName, sessionData, eyeTrackingData, audioChunks, onProgress) {
        let lastError;

        for (let attempt = 1; attempt <= config.maxRetries; attempt++) {
            try {
                const result = await uploadSession(sessionCode, gameName, sessionData, eyeTrackingData, audioChunks, onProgress);
                if (result.success) {
                    return result;
                }
                lastError = new Error(result.error);
            } catch (error) {
                lastError = error;
            }

            if (attempt < config.maxRetries) {
                const delay = config.retryDelayMs * Math.pow(2, attempt - 1);
                onProgress && onProgress(`Retry ${attempt}/${config.maxRetries} in ${delay / 1000}s...`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }

        return { success: false, error: lastError?.message || 'Upload failed after retries' };
    }

    // Public API
    return {
        /**
         * Upload session data to S3 via Lambda
         */
        upload: uploadWithRetry,

        /**
         * Get current upload status
         */
        getStatus: () => uploadStatus,

        /**
         * Check if upload is in progress
         */
        isUploading: () => uploadInProgress,

        /**
         * Set the API Gateway base URL (required before uploading)
         * @param {string} url - e.g., 'https://abc123.execute-api.us-east-1.amazonaws.com/prod'
         */
        setApiUrl: (url) => {
            config.apiBaseUrl = url.replace(/\/$/, ''); // Remove trailing slash
        },

        /**
         * Get the configured API URL
         */
        getApiUrl: () => config.apiBaseUrl
    };
})();
