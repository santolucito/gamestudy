/**
 * Lambda function: getStats
 *
 * Lists objects in the S3 bucket and returns aggregated session stats.
 *
 * Response: { totalSessions, totalFiles, sessionsByGame, sessionsByDate, mostRecentUpload }
 */

const { S3Client, ListObjectsV2Command } = require('@aws-sdk/client-s3');

const s3Client = new S3Client({ region: process.env.AWS_REGION || 'us-east-1' });
const BUCKET_NAME = process.env.BUCKET_NAME || 'gamestudy-data';

const ALLOWED_ORIGINS = [
    'https://marksantolucito.com',
    'https://www.marksantolucito.com',
    'http://marksantolucito.com',
    'http://www.marksantolucito.com',
    'https://r-papir.github.io',
    'http://r-papir.github.io'
];

function getCorsHeaders(event) {
    const origin = event.headers?.origin || event.headers?.Origin || '';
    return {
        'Access-Control-Allow-Origin': ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0],
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, OPTIONS'
    };
}

exports.handler = async (event) => {
    const corsHeaders = getCorsHeaders(event);

    // Handle CORS preflight
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 200,
            headers: corsHeaders,
            body: ''
        };
    }

    try {
        // List all objects under sessions/
        const objects = [];
        let continuationToken;

        do {
            const command = new ListObjectsV2Command({
                Bucket: BUCKET_NAME,
                Prefix: 'sessions/',
                ContinuationToken: continuationToken
            });
            const response = await s3Client.send(command);
            if (response.Contents) {
                objects.push(...response.Contents);
            }
            continuationToken = response.IsTruncated ? response.NextContinuationToken : undefined;
        } while (continuationToken);

        // Parse keys: sessions/{YYYY-MM-DD}/{code}-{game}/{filename}
        const sessions = new Set();
        const sessionsByGame = {};
        const sessionsByDate = {};
        let mostRecentUpload = null;

        for (const obj of objects) {
            const parts = obj.Key.split('/');
            // parts: ["sessions", date, sessionId, filename]
            if (parts.length < 4) continue;

            const date = parts[1];
            const sessionId = parts[2];

            sessions.add(`${date}/${sessionId}`);

            // Extract game name: sessionId is {code}-{game}
            const dashIndex = sessionId.indexOf('-');
            const gameName = dashIndex !== -1 ? sessionId.substring(dashIndex + 1) : sessionId;

            if (!sessionsByGame[gameName]) sessionsByGame[gameName] = new Set();
            sessionsByGame[gameName].add(`${date}/${sessionId}`);

            if (!sessionsByDate[date]) sessionsByDate[date] = new Set();
            sessionsByDate[date].add(`${date}/${sessionId}`);

            if (!mostRecentUpload || obj.LastModified > mostRecentUpload) {
                mostRecentUpload = obj.LastModified;
            }
        }

        // Convert sets to counts
        const sessionsByGameCounts = {};
        for (const [game, set] of Object.entries(sessionsByGame)) {
            sessionsByGameCounts[game] = set.size;
        }
        const sessionsByDateCounts = {};
        for (const [date, set] of Object.entries(sessionsByDate)) {
            sessionsByDateCounts[date] = set.size;
        }

        return {
            statusCode: 200,
            headers: corsHeaders,
            body: JSON.stringify({
                totalSessions: sessions.size,
                totalFiles: objects.length,
                sessionsByGame: sessionsByGameCounts,
                sessionsByDate: sessionsByDateCounts,
                mostRecentUpload: mostRecentUpload ? mostRecentUpload.toISOString() : null
            })
        };

    } catch (error) {
        console.error('getStats error:', error);
        return {
            statusCode: 500,
            headers: corsHeaders,
            body: JSON.stringify({ error: error.message })
        };
    }
};
