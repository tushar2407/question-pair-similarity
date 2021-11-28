from django.shortcuts import render
from main.utils import (
    uni_lr, uni_xg,
    bi_lr, bi_xg,
    tri_lr, tri_xg,
    bow_xg, 
    normalize_text, 
    # lstm_mlp
)
from question_similarity.settings import BASE_DIR, STATIC_URL, STATICFILES_DIRS
import os
# Create your views here.

def home(request):
    context = list(map(lambda x : f'{STATIC_URL}{x}', os.listdir(f'{STATICFILES_DIRS[0]}')))
    return render(request, 'main/home.html', context={'files':context})

def test(request):
    
    if request.method == 'POST':
        result = {}
        question_1 = normalize_text( request.POST.get('question_1', ''))
        question_2 = normalize_text( request.POST.get('question_2', ''))
        result['Unigram with Logistic Regression'] = uni_lr(question_1, question_2)[0]
        result['Unigram with XGBoost'] = uni_xg(question_1, question_2)[0]
        result['Bigram with Logistic Regression'] = bi_lr(question_1, question_2)[0]
        result['Bigram with XGBoost'] = bi_xg(question_1, question_2)[0]
        result['Trigram with Logistic Regression'] = tri_lr(question_1, question_2)[0]
        result['Trigram with XGBoost'] = tri_xg(question_1, question_2)[0]
        # result['LSTM with MLP'] = lstm_mlp(question_1, question_2)[0][0]
        return render(
            request, 
            'main/test.html', 
            context={
                'result' : result,
                'q1': request.POST.get('question_1', ''),
                'q2': request.POST.get('question_2', '')
            }
        )
    
    return render(request, 'main/test.html')