from tensorflow.keras.preprocessing.text import Tokenizer # 단어 인코딩 및 문장에서 벡터를 생성
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!',
    'Do you think my dog is amazing? hello',
    '안녕 안녕 한국어도 되나요 한번 봐주세요?'
]

tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>") 
tokenizer.fit_on_texts(sentences) #토큰화 
word_index = tokenizer.word_index #개수가 가장 많은 순서로 dict 형태로 만들어


sequences = tokenizer.texts_to_sequences(sentences)#시퀀스 텍스트 예를들어서 i love my dog 이라면 각 단어에 대한 word_index 를 표기함
padded = pad_sequences(sequences) #패딩 처리 앞부터 0으로 패딩 처리됨 
padded_post = pad_sequences(sequences, padding='post') #패딩 처리 뒤부터 0으로 패딩처리됨
padded_post5 = pad_sequences(sequences, padding='post', maxlen=5) #패딩 처리 <최고 긴것이 ..5 이상이면 앞에서부터 짤림>
padded_postpost5 = pad_sequences(sequences, padding='post', truncating='post', maxlen=5) #패딩 처리 <최고 긴것이 ..5 이상이면 뒤에껄 자름>

print("word_index: ", word_index)
print("sequences: ", sequences)
print("padded: ", padded)
print("padded_post: ", padded_post)
print("padded_post_maxlen5: ", padded_post5)
print("padded_post_post_maxlen5: ", padded_postpost5)

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
padded_test_data = pad_sequences(test_seq, maxlen=10)
print("test_deq: ",test_seq)
print("padded test data: ",padded_test_data)