from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!'
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)#토큰화
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)#시퀀스 텍스트 
padded = pad_sequences(sequences) #패딩 처리 
padded_post = pad_sequences(sequences, padding='post') #패딩 처리 
padded_post5 = pad_sequences(sequences, padding='post', maxlen=5) #패딩 처리 <최고 긴것이 ..5 이상이면 앞에서부터 짤림>
padded_postpost5 = pad_sequences(sequences, padding='post',truncating='post', maxlen=5) #패딩 처리 <최고 긴것이 ..5 이상이면 뒤에껄 자름>

print("word_index: ", word_index)
print("sequences: ", sequences)
print("padded: ", padded)
print("padded_post: ", padded_post)
print("padded_post5: ", padded_post5)
print("padded_pospost5: ", padded_postpost5)

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
padded_test_data = pad_sequences(test_seq, maxlen=10)
print("test_deq: ",test_seq)
print("padded test data: ",padded_test_data)