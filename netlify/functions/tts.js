const https = require('https');

const GEMINI_API_KEYS = [
    process.env.GEMINI_API_KEY || Buffer.from("QVEuQWI4Uk42S0xvOGQ0Qmw3MlpRZl9TNkFya3VDTHdMaUc4RVZEVkxwVnVXdUU1MEh3bGc=", "base64").toString("utf-8"),
    Buffer.from("QVEuQWI4Uk42SWZIRWx5eld0aHRfRGp6aW0xanNNRzdRbGtDQWFzbVVPRTNONFdrZE9FOFE=", "base64").toString("utf-8"),
    Buffer.from("QVEuQWI4Uk42SkUzN3R4czZVY0VRLWtaZ0FiejBYb0p5dnNMV1d5aTRwd2YwWVgwUV9jVXc=", "base64").toString("utf-8")
];

const VOICELAB_API_KEYS = [
    "vlk_e7R8sSc93z-qo11q2uauHiEhNeyzx4jeCbnJab4QfXM",  // Asosiy kalit
    "vlk_srm9l7U767RcNOLB8EehYML8HUsnt5KqLsJmia4-I1U",  // 1-Zaxira kalit
    "vlk_V7vy1tQvhAiGuU5y4vsTsJqcRRfVIEFnzpdJdIF5ods"   // 2-Zaxira kalit
];

const VOICELAB_VOICES = {
    "madina": "voice_4ef06e6b-2229-4023-ad66-c2a7a2b736ff",
    "amir": "voice_8bae02c6-d078-4757-9d62-0fce6da23cf7",
    "sardor": "voice_8bae02c6-d078-4757-9d62-0fce6da23cf7",
    "javohir": "voice_fa56cecf-9b0b-45ba-9ec3-494efb5b301f",
    "kamola": "voice_47260d60-5635-4fb0-904d-0f3c19c04ce6",
    "mohira": "voice_47260d60-5635-4fb0-904d-0f3c19c04ce6",
    "abdulhay": "voice_M-p5iz9z_kPWNm_C0cFqbC6X"
};

function requestGeminiTTS(apiKey, text) {
    return new Promise((resolve, reject) => {
        const payload = JSON.stringify({
            contents: [{ parts: [{ text: text }] }]
        });
        const url = `/v1beta/models/gemini-3.8-flash-lite-tts:generateContent?key=${apiKey}`;
        const req = https.request({
            hostname: 'generativelanguage.googleapis.com',
            path: url,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(payload)
            }
        }, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    try {
                        const json = JSON.parse(data);
                        const b64 = json.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
                        if (b64) {
                            resolve(Buffer.from(b64, 'base64'));
                        } else {
                            reject(new Error('No inline audio data'));
                        }
                    } catch(e) {
                        reject(e);
                    }
                } else {
                    reject(new Error(`Gemini TTS status ${res.statusCode}: ${data}`));
                }
            });
        });
        req.on('error', reject);
        req.write(payload);
        req.end();
    });
}

function requestTTSWithKey(apiKey, payload) {
    return new Promise((resolve, reject) => {
        const req = https.request({
            hostname: 'api.voicelab.uz',
            path: '/v1/tts',
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(payload)
            }
        }, (res) => {
            const chunks = [];
            res.on('data', chunk => chunks.push(chunk));
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    resolve(Buffer.concat(chunks));
                } else {
                    reject(new Error(`VoiceLab status ${res.statusCode}: ${Buffer.concat(chunks).toString()}`));
                }
            });
        });
        req.on('error', reject);
        req.write(payload);
        req.end();
    });
}

exports.handler = async function(event, context) {
    const params = event.queryStringParameters || {};
    const text = (params.text || '').trim();
    const voice = (params.voice || 'madina').toLowerCase();

    if (!text) {
        return { statusCode: 400, body: 'Text query parameter is required' };
    }

    // 1. Agar foydalanuvchi Gemini AI ovozini tanlagan bo'lsa
    if (voice === 'gemini' || voice === 'google') {
        for (let i = 0; i < GEMINI_API_KEYS.length; i++) {
            const apiKey = GEMINI_API_KEYS[i];
            try {
                const audioBuffer = await requestGeminiTTS(apiKey, text);
                return {
                    statusCode: 200,
                    headers: {
                        'Content-Type': 'audio/wav',
                        'Cache-Control': 'public, max-age=86400'
                    },
                    body: audioBuffer.toString('base64'),
                    isBase64Encoded: true
                };
            } catch (err) {
                console.warn(`Gemini TTS kalit [${i + 1}/${GEMINI_API_KEYS.length}] xatosi:`, err.message);
            }
        }
    }

    // 2. VoiceLab.uz milliy ovozlari (yoki zaxira)
    const voiceId = VOICELAB_VOICES[voice] || VOICELAB_VOICES["madina"];
    const payload = JSON.stringify({
        text: text,
        voice_id: voiceId,
        language: "uz"
    });

    let lastError = null;
    for (let i = 0; i < VOICELAB_API_KEYS.length; i++) {
        const apiKey = VOICELAB_API_KEYS[i];
        try {
            const audioBuffer = await requestTTSWithKey(apiKey, payload);
            return {
                statusCode: 200,
                headers: {
                    'Content-Type': 'audio/wav',
                    'Cache-Control': 'public, max-age=86400'
                },
                body: audioBuffer.toString('base64'),
                isBase64Encoded: true
            };
        } catch (err) {
            console.warn(`VoiceLab API kalit [${i + 1}/${VOICELAB_API_KEYS.length}] xatosi:`, err.message);
            lastError = err;
        }
    }

    return {
        statusCode: 502,
        body: JSON.stringify({ error: lastError ? lastError.message : 'All TTS keys failed' })
    };
};
