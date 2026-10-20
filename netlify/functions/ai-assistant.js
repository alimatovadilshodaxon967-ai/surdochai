const https = require('https');

const GEMINI_API_KEYS = [
    process.env.GEMINI_API_KEY || Buffer.from("QVEuQWI4Uk42S0xvOGQ0Qmw3MlpRZl9TNkFya3VDTHdMaUc4RVZEVkxwVnVXdUU1MEh3bGc=", "base64").toString("utf-8"),
    Buffer.from("QVEuQWI4Uk42SWZIRWx5eld0aHRfRGp6aW0xanNNRzdRbGtDQWFzbVVPRTNONFdrZE9FOFE=", "base64").toString("utf-8"),
    Buffer.from("QVEuQWI4Uk42SkUzN3R4czZVY0VRLWtaZ0FiejBYb0p5dnNMV1d5aTRwd2YwWVgwUV9jVXc=", "base64").toString("utf-8")
];

function requestGemini(apiKey, payload) {
    return new Promise((resolve, reject) => {
        const url = `/v1beta/models/gemini-flash-lite-latest:generateContent?key=${apiKey}`;
        const req = https.request({
            hostname: 'generativelanguage.googleapis.com',
            path: url,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(payload)
            }
        }, res => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    try {
                        resolve(JSON.parse(data));
                    } catch(e) {
                        reject(e);
                    }
                } else {
                    reject(new Error(`Gemini status ${res.statusCode}: ${data}`));
                }
            });
        });
        req.on('error', reject);
        req.write(payload);
        req.end();
    });
}

exports.handler = async function(event, context) {
    if (event.httpMethod !== 'POST') {
        return { statusCode: 405, body: 'Method Not Allowed' };
    }

    let body = {};
    try {
        body = JSON.parse(event.body || '{}');
    } catch(e) {}

    const question = (body.question || '').trim();
    if (!question) {
        return {
            statusCode: 400,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'error', message: 'Savol kiritilmadi' })
        };
    }

    const systemInstruction = "Siz Surdo AI loyihasining o'zbek tilidagi ovozli aqlli yordamchisisiz. Foydalanuvchi mikrofonda savol berdi. O'zbek tilida juda samimiy, madaniy, odobli va qisqa (1-2 jumlada) aniq javob bering. QAT'IY QOIDA: Javob ovozda o'qiladi, shuning uchun yulduzchalar (*), panjara (#), markdown, emoji yoki murakkab belgilar mutlaqo ishlatmang!";

    const payload = JSON.stringify({
        system_instruction: { parts: [{ text: systemInstruction }] },
        contents: [{ parts: [{ text: question }] }],
        generationConfig: { temperature: 0.3 }
    });

    for (let i = 0; i < GEMINI_API_KEYS.length; i++) {
        const apiKey = GEMINI_API_KEYS[i];
        try {
            const geminiRes = await requestGemini(apiKey, payload);
            let answer = geminiRes.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "";
            if ((answer.startsWith('"') && answer.endsWith('"')) || (answer.startsWith("'") && answer.endsWith("'"))) {
                answer = answer.slice(1, -1).trim();
            }

            return {
                statusCode: 200,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: 'success', question: question, answer: answer, model: 'Google Gemini' })
            };
        } catch(err) {
            console.warn(`Gemini Assistant kalit [${i + 1}/${GEMINI_API_KEYS.length}] xatosi:`, err.message);
        }
    }

    return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            status: 'success',
            question: question,
            answer: "Kechirasiz, hozir savolingizga javob bera olmadim. Iltimos, qaytadan urinib ko'ring.",
            model: 'Fallback'
        })
    };
};
