#%%
text = 'You say goodbye and I say hello.'
#%%
text = text.lower()
text = text.replace('.',' .')
text

# %%
words = text.split(' ')
words
# %%
word_to_id ={}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word
# %%
id_to_word[1]
# %%
word_to_id['hello']
# %%
import numpy as np
corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)
corpus
# %%
def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split(' ')
    word_to_id ={}
    id_to_word = {}
    
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id]= word
    
    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word
# %%
text = 'You say goodbye and I say hello.'
# %%
corpus, word_to_id, id_to_word = preprocess(text)
# %%
C = np.array([
    [0,1,0,0,0,0,0],
    [1,0,1,0,1,1,0],
    [0,1,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,1,0,0,0],
    [0,1,0,0,0,0,1],
    [0,0,0,0,0,1,0]
    ],dtype=np.int32)
# %%
def create_co_matrix(corpus,vocab_size,window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size),dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx -1
            right_idx = idx +1

            if left_idx >=0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] +=1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id,right_word_id]+=1
    
    return co_matrix

create_co_matrix(corpus=corpus, vocab_size=len(word_to_id))
# %%
def cos_similarity(x,y,eps=1e-8):
    nx = x/(np.sqrt(np.sum(x**2))+eps)
    ny = y/(np.sqrt(np.sum(y**2))+eps)
    return np.dot(nx,ny)

text = 'You say goodbye and I say hello.'
corpus, word_to_id ,id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus=corpus,vocab_size=vocab_size)
c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print(cos_similarity(c0,c1))
# %%
def most_similar(query, word_to_id, id_to_word,word_matrix, top=5):
    if query not in word_to_id:
        print(f'{query}을(를) 찾을 수 없습니다.')
    print('\n[query]'+query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)

    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i],query_vec)
    
    count = 0
    for i in (-1*similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word[i]}:{similarity[i]}')

        count+=1
        if count >= top:
            return
# %%
most_similar('you',word_to_id,id_to_word,C,top=5)
# %%
import numpy as np

c = np.array([[1,0,0,0,0,0,0]])
W = np.random.randn(7,3)
h = np.matmul(c,W)
print(h)
# %%
