# Surdo AI — Netlify Deployment Qo'llanmasi

Loyiha Netlify serverless platformasi uchun 100% moslashtirildi va barcha kerakli fayllar tayyorlandi:
- `dist/` — Optimizatsiya qilingan statik frontend (MediaPipe Edge AI, HTML5, CSS3, JS).
- `netlify/functions/tts.js` — VoiceLab.uz va Google Gemini neyron ovoz sintezatori (Serverless backend).
- `netlify/functions/ai-grammar.js` — Google Gemini sun'iy intellekt jumlalar generatori.
- `netlify/functions/ai-assistant.js` — Google Gemini mikrofon savol-javob AI yordamchisi.
- `netlify.toml` — Netlify routing, build va redirect konfiguratsiyasi.

---

## 🚀 1-usul: Netlify Drop (Eng oson va tez — 30 soniyada)

1. Brauzerda [app.netlify.com/drop](https://app.netlify.com/drop) sahifasini oching.
2. Agar hisobingiz bo'lmasa, Google yoki GitHub orqali kiring.
3. Loyiha ichidagi **`dist`** papkasini brauzerdagi maydonga sudrab tashlang (Drag & Drop).
4. Netlify darhol saytingizni internetga chiqaradi va sizga bepul `https://sizning-saytingiz.netlify.app` manzilini taqdim etadi!

---

## 🐙 2-usul: GitHub orqali Netlify'ga ulash (Tavsiya etiladi)

1. Loyihani GitHub repozitoriyangizga push qiling.
2. [app.netlify.com](https://app.netlify.com) saytiga kiring.
3. **"Add new site"** -> **"Import an existing project"** -> **"GitHub"** tugmasini bosing.
4. Repozitoriyangizni tanlang.
5. Netlify avtomatik tarzda `netlify.toml` faylini o'qiydi:
   - **Publish directory:** `dist`
   - **Functions directory:** `netlify/functions`
6. **"Deploy site"** tugmasini bosing. Barchasi avtomatik ravishda ishga tushadi!

---

## 💻 3-usul: Terminal orqali (Netlify CLI)

Loyihaning terminalida quyidagi buyruqni ishga tushiring:
```bash
npx netlify deploy --prod
```
- So'rovnomada **Publish directory** so'ralsa: `dist` deb yozing.
- Bir necha soniyada jonli havolangiz tayyor bo'ladi!
