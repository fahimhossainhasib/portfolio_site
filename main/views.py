from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404
from .models import BlogPost, Project
import markdown2
from django.core.files.storage import default_storage
from django.conf import settings
import os
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
from insightface.app import FaceAnalysis

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

def clipsniper_demo(request):
    context = {}
    if request.method == 'POST':
        video = request.FILES.get('video')
        image = request.FILES.get('image')
        if not video or not image:
            context['error'] = "Both video and image are required."
            return render(request, 'project_demo.html', context)
        if video and video.size > MAX_UPLOAD_SIZE:
            return HttpResponse("Video file is too large (max 50 MB allowed).", status=400)
        video_path = default_storage.save('temp/video.mp4', video)
        image_path = default_storage.save('temp/image.jpg', image)
        video_full = os.path.join(settings.MEDIA_ROOT, video_path)
        image_full = os.path.join(settings.MEDIA_ROOT, image_path)

        try:
            model_root = os.path.join("models")
            model = FaceAnalysis(name='buffalo_sc', root=model_root)
            model.prepare(ctx_id=-1)

            ref_embedding = extract_embeddings(image_full, model)

            cap = cv2.VideoCapture(video_full)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            match_timestamps = []

            for idx in range(total_frames):
                ret, frame = cap.read()        
                if not ret:
                    break
                faces = model.get(frame)
                for f in faces:
                    sim = cosine_similarity(ref_embedding, f.embedding)
                    if sim > 0.5:
                        timestamp = idx / fps
                        match_timestamps.append(timestamp)
                        break
            cap.release()

            segments = group_timestamps(match_timestamps, fps)
            video_clip = VideoFileClip(video_full)
            w, h = video_clip.size
            clips = [video_clip.subclip(start, end) for start, end in segments]
            final = concatenate_videoclips(clips).resize((w, h))

            output_path = os.path.join(settings.MEDIA_ROOT, "output", "result.mp4")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final.write_videofile(output_path, codec="libx264", audio=True, fps=fps)

            context['output_url'] = os.path.join(settings.MEDIA_URL, "output", "result.mp4")

        except Exception as e:
            context['error'] = str(e)

    return render(request, 'clipsniper_demo.html', context)