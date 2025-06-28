from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, get_object_or_404
from .models import BlogPost, Project
import markdown2
from django.core.files.storage import default_storage
from django.conf import settings
import os
import numpy as np
import cv2
import uuid
import json
import threading
import gc
import psutil
import subprocess
from moviepy.editor import VideoFileClip, concatenate_videoclips
from insightface.app import FaceAnalysis

model_root = os.path.join("models")
model = FaceAnalysis(name='buffalo_sc', root=model_root)
model.prepare(ctx_id=-1)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

def home_view(request):
    latest_posts = BlogPost.objects.order_by('-created_at')[:3]
    top_projects = Project.objects.all()[:3]
    context = {
        'top_projects': top_projects,
        'latest_posts': latest_posts,
    }
    return render(request, 'home.html', context)

def blog_detail(request, slug):
    post = get_object_or_404(BlogPost, slug=slug)
    content_html = markdown2.markdown(post.content, extras=["fenced-code-blocks", "tables", "break-on-newline"])
    return render(request, 'blog_detail.html', {'post': post, 'content_html': content_html})

def blog_list(request):
    posts = BlogPost.objects.order_by("-created_at")[:3]
    return render(request, "blog_list.html", {"posts": posts})

def project_detail(request, slug):
    project = get_object_or_404(Project, slug=slug)
    return render(request, 'project_detail.html', {'project': project})

def project_demo(request, slug):
    return render(request, 'project_demo.html', {'project_slug': slug})

def project_list(request):
    projects = Project.objects.all()
    return render(request, 'project_list.html', {'projects': projects})

def extract_embeddings(img_path, model):    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not load image.")    
    faces = model.get(img)
    if not faces or len(faces) == 0:
        raise ValueError("No face found in reference image.")
    return faces[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def group_timestamps(timestamps, fps, max_gap=0.5):
    if not timestamps:
        return []
    segments = []
    start = prev = timestamps[0]
    for t in timestamps[1:]:
        if t - prev <= max_gap:
            prev = t
        else:
            segments.append((start, prev + 1/fps))
            start = prev = t
    segments.append((start, prev + 1/fps))
    return segments

def process_video_job(job_id, video_path, image_path):
    status_path = os.path.join(settings.MEDIA_ROOT, "status", f"{job_id}.json")
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    try:
        ref_embedding = extract_embeddings(image_path, model)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        match_timestamps = []
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            faces = model.get(frame)
            for f in faces[:3]:
                sim = cosine_similarity(ref_embedding, f.embedding)
                if sim > 0.4:
                    timestamp = idx / fps
                    match_timestamps.append(timestamp)
                    print(f"[MATCH] Found at frame {idx}, time={timestamp:.2f}s, sim={sim:.2f}")
                    break
            if idx % 10 == 0:
                atomic_write_json({"done": False, "progress": int((idx/total_frames)*100)}, status_path)
        cap.release()
        segments = group_timestamps(match_timestamps, fps)
        segments = [(s, e) for s, e in segments if e - s > 0.1]
        if not segments:
            print("[WARNING] No matching segments found.")
            atomic_write_json({"done": False, "output_url": None, "message": "No matching clips found"}, status_path)
            return
        del cap
        del ref_embedding
        del faces
        gc.collect()
        print("Before videoclip")
        resized_path = os.path.join(settings.MEDIA_ROOT, "temp_clips", f"{job_id}_resized.mp4")
        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "scale=iw/2:ih/2",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "copy",
            resized_path
        ], check=True)
        video_clip = VideoFileClip(resized_path, audio=False)
        print("Before Try")
        try:
            w, h = video_clip.size
            print("Before subclips")
            clips = []
            target_height = 360
            print(f"[DEBUG] Total segments: {len(segments)}")
            for i, (start, end) in enumerate(segments):
                print(f"[DEBUG] Trying segment {i}: {start:.2f} → {end:.2f}")
                sub = video_clip.subclip(start, end)
                w, h = sub.size
                scale = target_height / h
                sub = sub.resize(height=target_height, width=int(w * scale))
                clips.append(sub)
                print(f"[DEBUG] Clip {i} added: {start:.2f}s → {end:.2f}s, resized")
            print("Before Concat")
            final = concatenate_videoclips(clips, method="chain")
            print("After Concat")
            output_dir = os.path.join(settings.MEDIA_ROOT, "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{job_id}.mp4")
            final.write_videofile(output_path, codec="libx264", audio=False, fps=fps)
            final.close()
            print("Video Written")
        finally:
            video_clip.close()
        atomic_write_json({"done": True, "output_url": f"{settings.MEDIA_URL}output/{job_id}.mp4"}, status_path)
        print("Changed JSON File")
        try:
            os.remove(resized_path)
        except Exception as e:
            print(f"Couldn't remove resized temp video: {e}")
        try:
            os.remove(video_path)
            os.remove(image_path)
        except Exception as cleanup_err:
            print(f"[WARNING] Failed to delete temp files for job {job_id}: {cleanup_err}")
        delete_after_delay(output_path, status_path, delay_seconds=3600)
    except Exception as e:
        with open(status_path, 'w') as f:
            json.dump({"done": False, "error": str(e)}, f)

def atomic_write_json(data, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
    os.replace(tmp_path, path)

def clipsniper_demo(request):
    job_id = request.GET.get("job_id")
    context = {}
    if request.method == 'POST':
        video = request.FILES.get('video')
        image = request.FILES.get('image')
        if not video or not image:
            context['error'] = "Both video and image are required."
            return render(request, 'project_demo.html', context)
        if video and video.size > MAX_UPLOAD_SIZE:
            return HttpResponse("Video file is too large (max 50 MB allowed).", status=400)
        job_id = str(uuid.uuid4())
        video_path = default_storage.save(f'temp/{job_id}_video.mp4', video)
        image_path = default_storage.save(f'temp/{job_id}_image.jpg', image)
        video_full = os.path.join(settings.MEDIA_ROOT, video_path)
        image_full = os.path.join(settings.MEDIA_ROOT, image_path)
        threading.Thread(target=process_video_job, args=(job_id, video_full, image_full)).start()
        return JsonResponse({"job_id": job_id})
    if job_id:
        status_path = os.path.join(settings.MEDIA_ROOT, "status", f"{job_id}.json")
        if os.path.exists(status_path):
            with open(status_path) as f:
                data = json.load(f)
            if data.get("done"):
                context["output_url"] = data["output_url"]
            else:
                context["error"] = "Video is not ready yet."
        else:
            context["error"] = "Invalid job ID."
        return render(request, 'clipsniper_demo.html', context)
    else:
        return render(request, 'project_demo.html', context)
    
def check_status(request):
    job_id = request.GET.get("job_id")
    if not job_id:
        return JsonResponse({"error": "job_id required"}, status=400)
    status_path = os.path.join(settings.MEDIA_ROOT, "status", f"{job_id}.json")
    if not os.path.exists(status_path):
        print("No Status Path")
        return JsonResponse({"done": False, "progress": 0})
    try:
        if os.path.getsize(status_path) == 0:
            print("Empty status file")
            return JsonResponse({"done": False, "progress": 0})
        with open(status_path) as f:
            data = json.load(f)
        return JsonResponse(data)
    except json.JSONDecodeError:
        print("[ERROR] Corrupted or incomplete JSON")
        return JsonResponse({"done": False, "progress": 0})
    except Exception as e:
        print(f"[ERROR] check_status failed: {e}")
        return JsonResponse({"error": str(e)}, status=500)

def delete_after_delay(file_path, status_path, delay_seconds=3600):
    def delete_file():
        try:
            if os.path.exists(file_path) and os.path.exists(status_path):
                # os.remove(file_path)
                # os.remove(status_path)
                print(f"[INFO] Deleted output: {file_path}")
        except Exception as e:
            print(f"[WARNING] Failed to delete output file {file_path}: {e}")
    threading.Timer(delay_seconds, delete_file).start()