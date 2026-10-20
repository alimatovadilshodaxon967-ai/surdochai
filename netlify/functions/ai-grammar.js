const https = require('https');

const GEMINI_API_KEYS = [
    process.env.GEMINI_API_KEY || Buffer.from("QVEuQWI4Uk42S0xvOGQ0Qmw3MlpRZl9TNkFya3VDTHdMaUc4RVZEVkxwVnVXdUU1MEh3bGc=", "base64").toString("utf-8"),
    Buffer.from("QVEuQWI4Uk42SWZIRWx5eld0aHRfRGp6aW0xanNNRzdRbGtDQWFzbVVPRTNONFdrZE9FOFE=", "base64").toString("utf-8"),
    Buffer.from("QVEuQWI4Uk42SkUzN3R4czZVY0VRLWtaZ0FiejBYb0p5dnNMV1d5aTRwd2YwWVgwUV9jVXc=", "base64").toString("utf-8")
];

function requestGeminiWithKey(apiKey, payload) {
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

    const words = Array.isArray(body.words) ? body.words : (body.words ? String(body.words).split(',') : []);
    const cleanWords = words.map(w => String(w).trim()).filter(Boolean);

    if (cleanWords.length === 0) {
        return {
            statusCode: 200,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                status: 'success',
                sentence: "Assalomu alaykum! Iltimos, avval imo-ishoralarni ko'rsating."
            })
        };
    }

    const wordsStr = cleanWords.join(', ');
    const systemInstruction = "Siz imo-ishoralarni nutqqa aylantiruvchi Surdo AI yordamchisisiz. Foydalanuvchi ko'rsatgan uzuq-yuluq imo-ishora so'zlaridan O'zbek tilida odobli, to'liq, ravon va madaniy bitta adabiy jumla tuzib bering. QAT'IY QOIDA: Faqat va faqat yakuniy gapni yozing! Kirish so'zlari, tushuntirishlar, qo'shtirnoq yoki '**' belgilarini aslo qo'shmang!";
    const prompt = `Imo-ishoralar so'zlari: ${wordsStr}. Til: O'zbek tili`;

    const payload = JSON.stringify({
        system_instruction: { parts: [{ text: systemInstruction }] },
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.25 }
    });

    for (let i = 0; i < GEMINI_API_KEYS.length; i++) {
        const apiKey = GEMINI_API_KEYS[i];
        try {
            const geminiRes = await requestGeminiWithKey(apiKey, payload);
            let sentence = geminiRes.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "";
            if ((sentence.startsWith('"') && sentence.endsWith('"')) || (sentence.startsWith("'") && sentence.endsWith("'"))) {
                sentence = sentence.slice(1, -1).trim();
            }

            return {
                statusCode: 200,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: 'success', sentence: sentence || wordsStr })
            };
        } catch(err) {
            console.warn(`Gemini API kalit [${i + 1}/${GEMINI_API_KEYS.length}] xatosi:`, err.message);
        }
    }

    return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'success', sentence: `Hurmat bilan: ${wordsStr}. Rahmat!` })
    };
};
