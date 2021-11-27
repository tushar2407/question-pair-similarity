from django.shortcuts import render
from main.utils import (
    uni_lr, uni_xg,
    bi_lr, bi_xg,
    tri_lr, tri_xg,
    bow_xg
)

# Create your views here.

def home(request):
    return render(request, 'main/home.html')

def test(request):
    result = {}
    
    if request.method == 'POST':
        question_1 = request.POST.get('question_1', '')
        question_2 = request.POST.get('question_2', '')
        print(question_2, question_1)
        result['uni_lr'] = uni_lr(question_1, question_2)
        result['uni_xg'] = uni_xg(question_1, question_2)
        result['bi_lr'] = bi_lr(question_1, question_2)
        result['bi_xg'] = bi_xg(question_1, question_2)
        result['tri_lr'] = tri_lr(question_1, question_2)
        result['tri_xg'] = tri_xg(question_1, question_2)
    
    return render(request, 'main/test.html', context=result)