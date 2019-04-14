import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

def parse_finish_line(line):
    return tf.py_func(data_parser, [line, 7], [tf.int32, tf.float32, tf.float32])

def data_parser(line, label_index):
    """ parser line content and generate idx, features, and gts """
    content = line.split('\t')
    label = np.float32(content[label_index].strip())
    feature_num = 5
    features = content[:feature_num + 1]
    features = map(lambda feature: np.float32(feature), features)
    idx = [0 if feature < 0 else feature for feature in features]
    features = [np.float32(0) if feature < 0 else np.float32(1) for feature in features]
    features = features[:feature_num]

    idx = idx[:feature_num]

    shifts = [0, 73974, 74370, 4197059, 5047367, 5047828, 5047833]
    idx = [idx[i] + shifts[i] for i in range(len(idx))]

    idx = map(lambda one_id: np.int32(one_id), idx)

    return idx, features, label

epoch = 10
batch_size = 4096
dataset = tf.data.TextLineDataset("../../data/train_data.txt")
dataset = dataset.map(parse_finish_line)
dataset = dataset.shuffle(buffer_size=300)
dataset = dataset.repeat(epoch)
dataset = dataset.batch(batch_size)

data_iterator = dataset.make_one_shot_iterator()
one_element = data_iterator.get_next()


embedding_size = 40
feat_num = 5047833

weights = dict()
weights["feature_embeddings"] = tf.get_variable(
        name='weights',
        dtype=tf.float32,
        initializer=tf.glorot_normal_initializer(),
        shape=[feat_num, embedding_size])

weights["weights_first_order"] = tf.get_variable(
        name='vectors',
        dtype=tf.float32,
        initializer=tf.glorot_normal_initializer(),
        shape=[feat_num, 1])

weights["bias"] = tf.get_variable(
    name='bias',
    dtype=tf.float32,
    initializer=tf.constant_initializer(0.0),
    shape=[1])

feature_idx = tf.placeholder('int32', [None, 5])
feature_val = tf.placeholder('float32', [None, 5, 1])
labels = tf.placeholder('float32', [None, 1])


weights_first_order = tf.nn.embedding_lookup(
    weights["weights_first_order"],
    feature_idx
)

embeddings = tf.nn.embedding_lookup(
    weights["feature_embeddings"],
    feature_idx
)
bias = weights['bias']
##first order
first_order = tf.multiply(feature_val, weights_first_order)
first_order = tf.reduce_sum(first_order, 2)
first_order = tf.reduce_sum(first_order, 1, keepdims=True)
##second order
### feature * embeddings
f_e_m = tf.multiply(feature_val, embeddings)
###  square(sum(feature * embedding))
f_e_m_sum = tf.reduce_sum(f_e_m, 1)
f_e_m_sum_square = tf.square(f_e_m_sum)
###  sum(square(feature * embedding))
f_e_m_square = tf.square(f_e_m)
f_e_m_square_sum = tf.reduce_sum(f_e_m_square, 1)

second_order = f_e_m_sum_square - f_e_m_square_sum

second_order = tf.reduce_sum(second_order, 1, keepdims=True)

##final objective function
logits =  first_order + second_order + bias
predicts = tf.sigmoid(logits)

##loss function
sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
sigmoid_loss = tf.reduce_mean(sigmoid_loss)
loss = sigmoid_loss

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cnt= 0
    while(True):
        try:
            cnt = cnt + 1
            feat_index, feat_val, lable = sess.run(one_element)
            feat_val = np.reshape(feat_val,newshape=[batch_size,5,1])
            lable = np.reshape(lable,newshape=[batch_size,1])
            #print feat_index,feat_val,lable
            _,t1 = sess.run([train_op,loss], feed_dict={feature_idx: feat_index, feature_val: feat_val, labels: lable})
            print cnt,t1
        except tf.errors.OutOfRangeError:
            break

        if(cnt%100 == 0):
            val = tf.data.TextLineDataset("../../data/test_data.txt")
            val_dataset = val.map(parse_finish_line)
            val_dataset = val_dataset.batch(batch_size)
            val_data_iterator = val_dataset.make_one_shot_iterator()
            result = []
            true_lable = []
            while(True):
                try:
                    feat_index, feat_val, lable = sess.run(val_data_iterator.get_next())
                    feat_val = np.reshape(feat_val, newshape=[-1, 5, 1])
                    true_lable = true_lable + list(lable)
                    lable = np.reshape(lable, newshape=[-1, 1])
                    pre = sess.run([predicts],feed_dict={feature_idx: feat_index, feature_val: feat_val, labels: lable})
                    result = result + list(np.reshape(pre, newshape=[-1]))
                except tf.errors.OutOfRangeError:
                    break
            print "test_auc:", roc_auc_score(true_lable, result)
            #saver.save(sess, "model/model.ckpt")