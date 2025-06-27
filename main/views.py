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
import time
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
            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            for f in faces[:3]:
                sim = cosine_similarity(ref_embedding, f.embedding)
                if sim > 0.5:
                    timestamp = idx / fps
                    match_timestamps.append(timestamp)
                    break
            if idx % 10 == 0:
                with open(status_path, 'w') as f:
                    json.dump({"done": False, "progress": int((idx/total_frames)*100)}, f)
        cap.release()

        segments = group_timestamps(match_timestamps, fps)
        video_clip = VideoFileClip(video_path)
        try:
            w, h = video_clip.size
            clips = [video_clip.subclip(start, end) for start, end in segments]
            final = concatenate_videoclips(clips).resize((w, h))
            output_dir = os.path.join(settings.MEDIA_ROOT, "output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{job_id}.mp4")
            final.write_videofile(output_path, codec="libx264", audio=True, fps=fps)
            final.close()
        finally:
            video_clip.close()
        with open(status_path, 'w') as f:
            json.dump({"done": True, "output_url": f"{settings.MEDIA_URL}output/{job_id}.mp4"}, f)
        try:
            os.remove(video_path)
            os.remove(image_path)
        except Exception as cleanup_err:
            print(f"[WARNING] Failed to delete temp files for job {job_id}: {cleanup_err}")
        delete_after_delay(output_path, status_path, delay_seconds=3600)
    except Exception as e:
        with open(status_path, 'w') as f:
            json.dump({"done": False, "error": str(e)}, f)


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
        return JsonResponse({"done": False})
    try:
        with open(status_path) as f:
            data = json.load(f)
        return JsonResponse(data)
    except Exception as e:
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