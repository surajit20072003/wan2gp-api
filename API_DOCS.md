# Wan2GP API Documentation

Welcome to the Wan2GP Multi-GPU API. Is API ko use karke aap server par chal rahe 3 GPUs par Text-to-Video aur Image-to-Video jobs submit, monitor, aur download kar sakte hain.

## 🔐 Authentication
Saare endpoints ab secure hain. Aapko har request ke headers mein apni API key bhejni hogi:
```http
X-API-Key: mypassword1234
```
> **Note:** Aap apni API key server ke `.env` file mein update kar sakte hain.

---

## 📡 API Endpoints

### 1. System Health & Status

**👉 API Health & Queue Status**
```bash
curl -H "X-API-Key: mypassword1234" http://localhost:8000/health
```

**👉 Detailed GPU Status**
GPUs free hain ya busy, ye check karne ke liye:
```bash
curl -H "X-API-Key: mypassword1234" http://localhost:8000/gpu_status
```

**👉 Queue System Stats**
```bash
curl -H "X-API-Key: mypassword1234" http://localhost:8000/queue
```

---

### 2. Available Models & LoRAs

**👉 List AI Models**
Text-to-Video aur Image-to-Video ke saare available templates/models dekhne ke liye:
```bash
curl -H "X-API-Key: mypassword1234" http://localhost:8000/models
```

**👉 List Available LoRAs**
```bash
curl -H "X-API-Key: mypassword1234" http://localhost:8000/loras
```

---

### 3. Media Upload (Image-to-Video & Audio)

Image-to-Video generate karne se pehle, aapko start frame image upload karni padegi. Uska ek `file_token` return hoga.

**👉 Upload Image/Audio via POST**
```bash
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: mypassword1234" \
  -F "file=@/path/to/your/image.jpg"
```
**Response aayega:**
```json
{
  "file_token": "a1b2c3d4.jpg",
  "original_filename": "image.jpg",
  "type": "image",
  "usage": {
    "image_start_token": "a1b2c3d4.jpg"
  }
}
```

---

### 4. Video Generation

**👉 Submit a Text-to-Video Job**
```bash
curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: mypassword1234" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic flying car over a cyberpunk city, highly detailed, 4k",
    "resolution": "1280x720",
    "video_length": 361,
    "steps": 8,
    "model": "ltx23_distilled_q6"
  }'
```
*Note: Agar aap `model` pass nahi karte, toh automatically `ltx23_distilled_q6` use hoga.*

**👉 Submit an Image-to-Video Job**
Image-to-video me hum wahi `file_token` use karenge jo upload me mila tha.
```bash
curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: mypassword1234" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The car suddenly accelerates forward into the neon lights",
    "image_start_token": "a1b2c3d4.jpg",
    "model": "ltx23_distilled_q6"
  }'
```
**Response aayega:**
```json
{
  "job_id": "job_1700000000_a1b2c3d4",
  "status": "queued",
  "queue_position": 1,
  "estimated_wait_minutes": 4.0
}
```

---

### 5. Job Tracking & Download

**👉 Check Job Status**
```bash
curl -H "X-API-Key: mypassword1234" http://localhost:8000/status/job_1700000000_a1b2c3d4
```
Job status: `queued`, `processing`, `running`, `completed`, `failed` aayega.

**👉 Download Finished Video**
Agar status `completed` aata hai, toh video download kar lijiye:
```bash
curl -H "X-API-Key: mypassword1234" http://localhost:8000/download/job_1700000000_a1b2c3d4 --output my_video.mp4
```

---

### 6. Job Management

**👉 List Recent Jobs**
Pichle generated jobs aur unka status dekhne ke liye:
```bash
curl -H "X-API-Key: mypassword1234" "http://localhost:8000/jobs/list?limit=20"
```

**👉 Retry a Failed Job**
```bash
curl -X POST -H "X-API-Key: mypassword1234" http://localhost:8000/retry/job_1700000000_a1b2c3d4
```

**👉 Delete Job History**
```bash
curl -X DELETE -H "X-API-Key: mypassword1234" http://localhost:8000/jobs/job_1700000000_a1b2c3d4
```
