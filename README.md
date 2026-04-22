# Local Development
## Init environment
```
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```
Set your Hugging Face token in `.env`:
```
HF_TOKEN=your_hf_token_here
# Keep cache-only loading enabled (default)
HF_LOCAL_FILES_ONLY=1
```
## Start FastAPI server
```
python -m uvicorn finbot.main:app --reload --app-dir src
```
## Frontend porting
```
cd frontend
python -m http.server 5500
```