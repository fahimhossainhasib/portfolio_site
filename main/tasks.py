import threading
import uuid
import json
import os
from django.conf import settings
from .views import clipsniper_demo

STATUS_FILE = os.path.join(settings.MEDIA_ROOT, 'task_status.json')

def load_status():
    if not os.path.exists(STATUS_FILE):
        return {}
    with open(STATUS_FILE, 'r') as f:
        return json.load(f)

def save_status(status_dict):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status_dict, f)

def update_task(task_id, data):
    status = load_status()
    status[task_id] = data
    save_status(status)

def background_process(task_id, image_path, video_path):
    try:
        update_task(task_id, {"status": "processing"})
        output_url = clipsniper_demo(image_path, video_path)
        update_task(task_id, {"status": "done", "output": output_url})
    except Exception as e:
        update_task(task_id, {"status": "error", "message": str(e)})

def start_task(image_path, video_path):
    task_id = str(uuid.uuid4())
    update_task(task_id, {"status": "queued"})
    threading.Thread(target=background_process, args=(task_id, image_path, video_path)).start()
    return task_id
