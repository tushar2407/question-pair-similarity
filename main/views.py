from django.shortcuts import render

# Create your views here.

def home(request):
    return render(request, 'main/home.html')

def test(request):
    result = {}
    
    if request.method == 'POST':
        question_1 = request.POST.get('question_1', '')
        question_2 = request.POST.get('question_2', '')
        print(question_2, question_1)
    
    return render(request, 'main/test.html', context=result)