# 📄 AI Document Intelligence

**Upload PDFs, ask questions, get answers with page-level citations (RAG)**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-frontend-black?logo=nextdotjs&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-blueviolet)

Upload a document, ask a question in plain English, and get an answer grounded in **your document**,
with the **source page** shown so you can verify it.

---

## 📸 Demo
<!-- ![demo](docs/demo.png) or a short GIF -->

---

## 🧠 How it works (RAG pipeline)

```
📄 PDF  →  ✂️ Chunking  →  🔢 Embeddings  →  🗄️ FAISS index
                                                   │
❓ Question  →  🔢 Embed question  →  🔍 Top-k similar chunks
                                                   │
                       🤖 LLM answers using ONLY those chunks
                                                   │
                         ✅ Answer + 📌 source pages
```

---

## ✨ Features

Tick only what is working in your repo.

| Status | Feature |
|---|---|
| - [ ] | 📤 PDF upload and text extraction |
| - [ ] | ✂️ Chunking with overlap |
| - [ ] | 🔢 Embeddings + FAISS vector search |
| - [ ] | 🤖 LLM Q&A with **source/page citations** |
| - [ ] | 💬 Chat interface |
| - [ ] | 📚 Multi-document support |
| - [ ] | 🕘 Chat history (MongoDB) |
| - [ ] | 🔐 Authentication (Auth.js / NextAuth) |
| - [ ] | 🐳 Docker setup |
| - [ ] | 🚀 Deployed demo |
| - [ ] | 📊 RAG evaluation |

---

## 🧰 Tech stack

| Layer | Tools |
|---|---|
| 🧠 AI | OpenAI API, LangChain, FAISS |
| 📄 Documents | PyPDF |
| ⚙️ Backend | Python, FastAPI |
| 🖥️ Frontend | Next.js, React, Tailwind CSS |
| 🗃️ Database | MongoDB |
| 🔐 Auth | Auth.js / NextAuth |
| 🚢 DevOps | Docker, Git/GitHub, Vercel + backend hosting |

*(Delete the rows for tools you don't use yet.)*

---

## 🚀 Quick start

### 1️⃣ Clone
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
```
Open `.env` and add your keys (see **Environment variables** below), then:
```bash
uvicorn app.main:app --reload
```
API docs: http://localhost:8000/docs

### 3️⃣ Frontend
```bash
cd frontend
npm install
npm run dev
```
App: http://localhost:3000

> 🧩 Early milestone? If you only have the CLI version, run `python main.py` from `backend/`.

---

## 🔑 Environment variables

Create `.env` from `.env.example`. **Never commit `.env`.**

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM and embeddings |
| `MONGODB_URI` | Chat history storage |
| `NEXTAUTH_SECRET` | Auth session secret |
| `NEXTAUTH_URL` | Frontend URL |

*(Keep only the variables your project uses.)*

---

## 💡 Example

```
You:  What is the notice period in this contract?
Bot:  The notice period is 30 days.
      📌 Source: contract.pdf, page 4
```
<!-- Replace with a real example from your own demo. -->

---

## 🗂️ Project structure

```
📦 ai-document-intelligence
├── 📁 backend/
│   ├── app/
│   │   ├── main.py            ← FastAPI entry
│   │   ├── routes/            ← upload, chat endpoints
│   │   ├── services/          ← pdf loading, chunking, embeddings, retrieval
│   │   └── db/                ← MongoDB access
│   ├── requirements.txt
│   └── .env.example
├── 📁 frontend/               ← Next.js app (chat UI, upload, auth)
├── 🐳 docker-compose.yml
├── 📄 .gitignore
└── 📄 README.md
```
*(Adjust to match your real folders.)*

---

## 📊 Evaluation

> Fill in only results you actually measured.

| Item | Value |
|---|---|
| Test documents | `<number, type>` |
| Questions evaluated | `<number>` |
| Answer correctness | `<X% or method used>` |
| Citation accuracy | `<X%>` |
| Avg. response time | `<X s>` |

Describe your method in one line, e.g. how you judged correctness and what questions you used.

---

## ⚠️ Limitations
- Answers depend on retrieval quality; unclear or scanned (image-only) PDFs may extract poorly.
- The LLM can still make mistakes, so check the cited page.
- Needs an OpenAI API key; usage costs apply.

---

## 🛣️ Roadmap
- [ ] Better chunking and re-ranking
- [ ] Support for more file types (DOCX, TXT)
- [ ] Streaming responses
- [ ] Rate limiting and usage tracking

---

## 🔒 Security notes
- Keep API keys in `.env`, never in code or Git.
- Uploaded documents are stored at `<path/DB>`; delete them if they contain sensitive data.
