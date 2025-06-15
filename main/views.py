from django.shortcuts import render, get_object_or_404
from .models import BlogPost
import markdown2

def home_view(request):
    latest_posts = BlogPost.objects.order_by('-created_at')[:3]
    return render(request, 'home.html', {'latest_posts': latest_posts})

def blog_detail(request, slug):
    post = get_object_or_404(BlogPost, slug=slug)
    content_html = markdown2.markdown(post.content, extras=["fenced-code-blocks", "tables", "break-on-newline"])
    return render(request, 'blog_detail.html', {'post': post, 'content_html': content_html})

def blog_list(request):
    posts = BlogPost.objects.order_by("-created_at")[:3]
    return render(request, "blog_list.html", {"posts": posts})