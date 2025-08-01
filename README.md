# Voice HR Bot

A full-stack AI-powered HR interview simulator with voice input, built with React (Vite) frontend and Django backend.

---

## Features

- 🎤 **Voice Input:** Practice interviews using your microphone.
- 🤖 **AI Interviewer:** Realistic, role-specific interview questions and feedback.
- 🖥️ **Modern UI:** Responsive, glassmorphic design.
- 🌐 **Deployable:** Easily host frontend (Netlify) and backend (Render/Heroku).

---

## Project Structure

```
voice-hr-bot/
  frontend/   # React + Vite app
  backend/    # Django API
```

---

## Getting Started (Local)

### **Frontend**

```sh
cd frontend
npm install
npm run dev
```

### **Backend**

```sh
cd backend
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

---

## Deployment

### **Frontend (Netlify)**

1. Push `frontend` to GitHub.
2. Import to Netlify, set build command: `npm run build`, publish directory: `dist`.

### **Backend (Render.com)**

1. Push `backend` to GitHub.
2. Create new Web Service on Render.
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn interviewsim.wsgi`
5. Add environment variables from `.env`.

---

## Environment Variables

Create a `.env` file in `backend/`:

```
GEMINI_API_KEY=your-google-gemini-api-key
```

---

## API

- **POST** `/api/chat/`
  - `{ "message": "Your answer", "role": "Data Scientist" }`
  - Returns: `{ "reply": "AI response" }`

---

## License

MIT
