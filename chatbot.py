import numpy as np
import tensorflow as tf
import re
import time # to measure training time.

####### Data Preprocessing ########
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

#creating a dctionary that maps each line and it's id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

#creating a list of all movie_conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))
#getting the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(0, len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

#cleaning the text
def cleanText(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-(){}\"@#$%^&*<>+=/:;.?,]", "", text)
    return text

#cleaning questions
clean_questions = []
for q in questions:
    clean_questions.append(cleanText(q))

clean_answers = []
for a in answers:
    clean_answers.append(cleanText(a))

#to remove the non frequent words

word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word]+=1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word]+=1

#tokenization
#generally 5% is the threshod
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if(count >= threshold):
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if(count >= threshold):
        answerswords2int[word] = word_number
        word_number += 1

#adding the tokens
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int)+1

for token in tokens:
    answerswords2int[token] = len(answerswords2int)+1

# inverse of the dictionary
answersints2words = {w_i:w for w,w_i in answerswords2int.items()}

#addind EOS token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

#translating the questions and answers into integers
#and replacing the words that were filtered out by <OUT>

questions2int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if(word not in questionswords2int):
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions2int.append(ints)

answers2int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if(word not in answerswords2int):
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers2int.append(ints)

#sorting questions and answers by the length of the questions to speed up the training
sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1,26):
    for i in enumerate(questions2int):
        if (len(i[1]) == length):
            sorted_clean_questions.append(questions2int[i[0]])
            sorted_clean_answers.append(answers2int[i[0]])

#Building the SEQ2SEQ model

#create place holders for the inputs and the targets
#tensors are more advanced np arrays. np arrays are more advanced variables
#in tensorflow each variable is stored in a tensor

 #create placeholders for inputs, targets
 #create placeholders for learning rate and other hyper parameters

def model_inputs():
     inputs = tf.placeholder(tf.int32, [None, None], name = 'input') #None is used to tell that the dimension is 2x2
     targets = tf.placeholder(tf.int32, [None, None], name = 'target') #None is used to tell that the dimension is 2x2
     lr = tf.placeholder(tf.float32, name = 'learning_rate') # keep prob hyperparameter = used to control the (drop out rate) = the rate of the nuerons you chose to overwrite during one iteration in the training. Usually the drop out rate is 20% of nuerons.
     keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
     return inputs,targets,lr,keep_prob

#processing the targets
#learn why from video

#the decoder needs the answers ina batch. not as a single input
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], axis = 1)
    return preprocessed_targets

#creating the encoding layer and decoding layers
#each layer is a LSTM RNN (Gated recurrent unit is something like LSTM)


#encoder

#rnn_size is the number of input tensors of the encoder rnn layer
#list of the length of each question a batch.
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, cell_bw = encoder_cell, sequence_length = sequence_length, inputs = rnn_inputs, dtype = tf.float32)
    return encoder_state

#decoder

#decode the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input , sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_state = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, training_decoder_function, decoder_embedded_input, sequence_length, scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


#decoding the test/validation set

#for the test set data. this is not part of the training data
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximun_length_answers, num_words_answers, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_state = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function, encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function,decoder_embedded_matrix, sos_id, eos_id, maximun_length_answers, num_words_answers, name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, test_decoder_function, scope = decoding_scope)
    return test_predictions

#decoder rnn usinf decoder trainig set and decoder testing set

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words_answers, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers) # lstm with one layer. we need stacked one. we need multi rnn cells to stack several lstm layers with dropout applied
        weigths = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x, num_words_answers, None, scope = decoding_scope, weigths_initializer = weigths, biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, word2int['<SOS>'], word2int['<EOS>'], sequence_length-1, num_words_answers, decoding_scope, output_function, keep_prob, batch_size)
        return training_predictions, test_predictions

#assembling all parts
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input  = tf.contrib.layers.embed_sequence(inputs, answers_num_words + 1, encoder_embedding_size, initializer = tf.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, questions_num_words, sequence_length, rnn_size, num_layers, questionswords2int, keep_prob, batch_size)
    return training_predictions, test_predictions

############# Training #############

#setting the hyperparameters

epochs = 100 #the whole process of getting the batches of inputs.
batch_size = 64  #the batch_size
rnn_size = 512 #size of rnn
num_layers = 3 #num of layers in encoder and decoder
encoding_embedding_size = 512  #num of columns in decoder_embeddings_matrix, each line corresponds to each token in the corpus of questions
decoding_embedding_size = 512  #num of columns in decoder_embeddings_matrix, each line corresponds to each token in the corpus of questions
learning_rate = 0.01 #speed at which the bot learn to speak
learning_rate_decay = 0.9 #how much the lr is reduced while training
minimum_learning_rate = 0.0001 #threshold value for lr
keep_probability = 0.5

#definig a session
tf.reset_default_graph()
session  = tf.InteractiveSession()
#loading model inputs
inputs,targets, lr, keep_prob = model_inputs()

#setting the sequence_length
sequence_length = tf.placeholder_with_default(25, None, name = 'Sequence_Length') # we will not use question/answers in training with more than 25 words

#getting the shape of input tensors
input_shape = tf.shape(inputs)

#getting the trainng and test predictions

training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]), targets, keep_prob, batch_size, sequence_length, len(answerswords2int), len(questionswords2int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, questionswords2int)

#setting the loss error, the optimizer and gradient clipping(avoid vanishing gradient descent)
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, targets, tf.ones([input_shape[0], sequence_length]))
    optimiser = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradient(loss_error)
    clipped_gradients = [(tf.clpi_by_value(grad_tensor, -5., 5), grad_variable)  for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_cipping = optimizer.apply_gradients(clipped_gradients)

#padding the sequence with <PAD> token to make the length of questiona and answer
#Question: ['who', 'are', 'you'] becomes
#answer: [<SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>]
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (maximum_sequence_length - len(sequence)) for sequence in batch_of_sequence]

#splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch

#splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split : ]
training_answers = sorted_clean_answers[training_validation_split : ]
validation_questions = sorted_clean_questions[ : training_validation_split]
validation_answers = sorted_clean_answers[ : training_validation_split]

# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_trainig_loss_error = 0

list_validation_loss_error = []
early_stopping_check = 0
early_stopping_threshold = 1000#too high
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variable_initializer())

for epoch in range(0, epochs):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch, targets: padded_answers_in_batch, lr : learning_rate, sequence_length: padded_answers_in_batch.shape[1], keep_prob: keep_probability})
        toatal_training_loss_error += batch_training_loss_error
        ending_time = tine.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print("Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training time on 100 batches: {:>d} seconds".format(epoch, epochs, batch_index, len(training_questions) // batch_size, total_trainig_loss_error / batch_index_check_training_loss, int(batch_time * batch_index_check_training_loss)))
            toatal_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch, targets: padded_answers_in_batch, lr : learning_rate, sequence_length: padded_answers_in_batch.shape[1], keep_prob: 1})
                toatal_validation_loss_error += batch_validation_loss_error
            ending_time = tine.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print("Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds".format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < minimum_learning_rate:
                learning_rate = minimum_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print("I speak better now")
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to process more")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_threshold:
                    break
    if early_stopping_check == early_stopping_threshold:
        print("My apologies, I cannot speak better anymore.")
        break
print ("Over")


############# Testing #############

#get weights from trainig
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variable_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

#converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = cleanText(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]

#setting up the chat
while (True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (20 - len(question))
    #nn will accept input on;y in batches. so we have to intiialize some empty questions
    fake_batch = np.zeros((batch_size), 20)
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    #post processing
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2words[i] == 'i':
            token = 'I'
        elif answersints2words[i] == '<EOS>':
            token = '.'
        elif answersints2words[i] == '<OUT>':
            token = 'out'
        elif:
            token = ' ' + answersints2words[i]
        answer += token
        if token == '.':
            break
    print("Chatbot: " + answer)
