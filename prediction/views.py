from django.shortcuts import render
from django.http import JsonResponse
from .forms import UploadImageForm
from .predict import predict_image
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

def predict_view(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            path = default_storage.save('uploads/' + image.name, ContentFile(image.read()))
            image_url = default_storage.url(path)

            label, confidence = predict_image(default_storage.path(path))

            return JsonResponse({
                'image_url': image_url,
                'label': label,
                'confidence': float(confidence) if confidence else 0.0
            })

        return JsonResponse({'error': 'Form tidak valid'}, status=400)

    # Untuk GET, tampilkan template biasa
    return render(request, 'index.html')
