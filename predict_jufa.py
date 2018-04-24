import tensorflow as tf

import numpy as np



filepath="/media/hadoop/文档/rpn3/yuliao.txt"
file = open(filepath) 
filename=[]
word_to_id={}
max_len=0
dic=[]
dic_to_id={}
sentence_list=[]
dic.append("PADID") #補0
dic.append("BID")   #開始
dic.append("EID")   #結束
#file2=str.replace(file3,"\n",'')
for line in file:
    
  for word in line:
    #word=line.split("")
    print(word)
    for w in word: 
        if w not in dic:
            dic.append(w)
  sentence_list.append(line)
print(dic)
num=0
for w in dic:
     
     word_to_id[w]=num
     num=num+1
print("word_to_id:",word_to_id)
#print(sentence_list)
sentence_to_id=[]
for s in  sentence_list:
     sent=[]
     sent.append(word_to_id["BID"])
     for wd in s:
         sent.append(word_to_id[wd])
     sent.append(word_to_id["EID"])
     sentence_to_id.append(sent)
for s in sentence_to_id:
     print(s)

learning_rate=0.1

#句子長度
length=220
batch_size=1
encoder_inputs=tf.placeholder(dtype=tf.int32,shape=[length,batch_size])
decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[length,batch_size])

#logits=tf.placeholder(dtype=tf.float32,shape=[10])
targets=tf.placeholder(dtype=tf.int32,shape=[length,batch_size])
weights=tf.placeholder(dtype=tf.float32,shape=[length,batch_size])


train_e_in=[]
train_d_in=[]
train_w=np.ones(shape=[length,batch_size],dtype=np.float32)

num_encoder_symbols=len(dic)
num_decoder_symbols=len(dic)
embedding_size=5
cell=tf.nn.rnn_cell.BasicLSTMCell(5)

#补零
count=0

weight_list=[]
for sent in   sentence_to_id:
     print(len(sent))
     d=length-len(sent)
     #s = (1,d)
     sent2=[]
     #weight=[]
     if d>0:
        for s in sent:
            sent2.append(s)
           # weight.append(1.0)
        for i in range(d): 
            sent2.append(0)
            #weight.append(0.0)
     else:
        sent2=sent
       # for s in sent:
           # weight.append(1.0)

     print(sent2)
     #if count%2==0:
     st_i=[]
     for w in sent2:
         if w!=word_to_id[" "]:
            st_i.append(w)
     d=length-len(st_i)
     if d>0:
        for j in range(d):
            st_i.append(0)
             
     train_e_in.append(st_i)
     #else:
     train_d_in.append(sent2)
       #weight_list.append(weight)
     count=count+1
     

print("train_e_in:",train_e_in)

def seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,embedding_size):
	encoder_inputs = tf.unstack(encoder_inputs, axis=0)
	decoder_inputs = tf.unstack(decoder_inputs, axis=0)
	results,states=tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    encoder_inputs,
    decoder_inputs,
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size,
    output_projection=None,
    feed_previous=False,
    dtype=None,
    scope=None
)
	return results

def get_loss(logits,targets,weights):
	loss=tf.contrib.seq2seq.sequence_loss(
		logits,
		targets=targets,
		weights=weights
	)
	return loss

results=seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,embedding_size)
logits=tf.stack(results,axis=0)
print(logits)
loss=get_loss(logits,targets,weights)

opt = tf.train.GradientDescentOptimizer(learning_rate)
    
update = opt.apply_gradients(opt.compute_gradients(loss))
    
#saver = tf.train.Saver(tf.global_variables())

#saver = tf.train.Saver(max_to_keep=3)

for din in train_d_in:
    weight=[]    
    for d in din:
        if d !=0:
            weight.append(1.0)
        else:
            weight.append(0.0)

    weight_list.append(weight)
    print("din:",din)
    print("weight",weight)


sess=tf.Session() 
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('Model/model.ckpt-240.meta')
saver.restore(sess,tf.train.latest_checkpoint('Model/'))

for i in range(1000):
     input_seq = input('请输入：') 


     seq_list2=jieba.cut(input_seq.strip())

     sentence=" ".join(seq_list2)
     word=sentence.split(" ")
     s_to_id=[]
     s_to_id.append(word_to_id["BID"])

     for w in word: 
          print(w)
          s_to_id.append(word_to_id[w])
     print(s_to_id)
     s_to_id.append(word_to_id["EID"])
     d=length-len(s_to_id)
     if d>0:
        for di in range(d):
           s_to_id.append(0)
     print(s_to_id)
     train_d=sess.run(tf.expand_dims(s_to_id,-1))
     train_e=sess.run(tf.expand_dims(train_e_in[0],-1))
     results_value=sess.run(results,feed_dict={encoder_inputs:train_e,decoder_inputs:train_d})
     outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in results_value]
     print(outputs_seq)
     cn=[]
     for o in outputs_seq:
        word=dic_to_id[o]
        cn.append(word)
     print("回復:",cn)


