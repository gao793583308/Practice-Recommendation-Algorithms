import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

def parse_finish_line(line):
    return tf.py_func(data_parser, [line, 6], [tf.int32, tf.float32, tf.float32])

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

def build_cin(nn_input):
    final_len = 0
    final_result = []
    cross_layer_size = [5,5,5]
    hidden_nn_layers = []
    hidden_nn_layers.append(nn_input)
    split_tensor0 = tf.split(hidden_nn_layers[0],40*[1],2)
    with tf.variable_scope("exfm_part") as scope:
        for idx,layer_size in enumerate(cross_layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1],40*[1],2)
            dot_result_m = tf.matmul(split_tensor0,split_tensor,transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m,shape=[40,-1,25])
            dot_result = tf.transpose(dot_result_o,perm=[1,0,2])
            filters = tf.get_variable(name="f_"+str(idx),shape=[1,25,layer_size],dtype=tf.float32)
            curr_out = tf.nn.conv1d(dot_result,filters=filters,stride=1,padding='VALID')

            b = tf.get_variable(name="f_b"+str(idx),
                                shape=[layer_size],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            curr_out = tf.nn.bias_add(curr_out,b)
            curr_out = tf.nn.relu(curr_out)
            curr_out = tf.transpose(curr_out,perm=[0,2,1])

            next_hidden = curr_out
            direct_connect = curr_out
            final_len = final_len + layer_size
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result,axis=1)
        result = tf.reduce_sum(result,-1)
        w_nn_output = tf.get_variable(name='w_nn_out',
                                       shape=[final_len,1],
                                       dtype=tf.float32)
        b_nn_output = tf.get_variable(name='b_nn_output',
                                      shape=[1],
                                      dtype=tf.float32,
                                      initializer=tf.zeros_initializer())
        exFM_out = tf.nn.xw_plus_b(result,w_nn_output,b_nn_output)
    return exFM_out

def build_fm(nn_input,feature_idx):
    with tf.variable_scope("fm_part") as scope:
        weights = dict()
        weights['weights_first_order'] = tf.get_variable(
            name='vectors',
            dtype=tf.float32,
            initializer=tf.glorot_normal_initializer(),
            shape=[feat_num,1]
        )
        weights['bias'] = tf.get_variable(
            name='bias',
            dtype=tf.float32,
            initializer=tf.glorot_normal_initializer(),
            shape=[1]
        )
        #---------------- First order term -----------------------------------
        y_first_order = tf.nn.embedding_lookup(weights['weights_first_order'],feature_idx)
        y_first_order = tf.reduce_sum(y_first_order,2)
        y_first_order = tf.reduce_sum(y_first_order,1,keep_dims=True)
        #---------------- Second order term-----------------------------------
        f_e_m_sum = tf.reduce_sum(nn_input,1)
        f_e_m_sum_square = tf.square(f_e_m_sum)
        f_e_m_square = tf.square(nn_input)
        f_e_m_square_sum = tf.reduce_sum(f_e_m_square,1)
        y_second_order = f_e_m_sum_square-f_e_m_square_sum
        y_second_order = tf.reduce_sum(y_second_order,1,keep_dims=True)
        return y_second_order+y_first_order+weights['bias']

def build_deep(nn_input):
    with tf.variable_scope("deep_part") as scope:
        weights = dict()
        deep_layers = [32,32]
        weights["layer_0"] = tf.get_variable(
            name='layer_0',
            dtype=tf.float32,
            initializer=tf.glorot_normal_initializer(),
            shape=[5*embedding_size,deep_layers[0]]
        )
        weights['bias_0'] = tf.get_variable(
            name='bias_0',
            dtype=tf.float32,
            initializer=tf.glorot_normal_initializer(),
            shape=[1,deep_layers[0]]
        )
        for i in range(1,len(deep_layers)):
            weights["layer_%d"%i] = tf.get_variable(
                name="layer_%d"%i,
                dtype=tf.float32,
                initializer=tf.glorot_normal_initializer(),
                shape=[deep_layers[i-1],deep_layers[i]]
            )
            weights["bias_%d"%i] = tf.get_variable(
                name='bias',
                dtype=tf.float32,
                initializer=tf.glorot_normal_initializer(),
                shape=[1,deep_layers[i]]
            )

        y_deep = tf.reshape(nn_input,shape=[-1,5*embedding_size])
        for i in range(len(deep_layers)):
            y_deep = tf.add(tf.matmul(y_deep,weights["layer_%d"%i]),weights["bias_%d"%i])
            y_deep = tf.nn.relu(y_deep)

        weights['concat_projection'] = tf.get_variable(
            name='concat_projection',
            dtype=tf.float32,
            initializer=tf.glorot_normal_initializer(),
            shape=[deep_layers[-1],1]
        )
        weights['concat_bias'] = tf.get_variable(
            name='concat_bias',
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),
            shape=[1]
        )
        deep_out = tf.add(tf.matmul(y_deep,weights['concat_projection']),weights['concat_bias'])
        return deep_out


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
#----------------- ini FM weights ---------------------
weights["feature_embeddings"] = tf.get_variable(
        name='weights',
        dtype=tf.float32,
        initializer=tf.glorot_normal_initializer(),
        shape=[feat_num, embedding_size])
#---------------------------------------------------------
feature_idx = tf.placeholder('int32', [None, 5])
feature_val = tf.placeholder('float32', [None, 5, 1])
labels = tf.placeholder('float32', [None, 1])
embeddings = tf.nn.embedding_lookup(
    weights["feature_embeddings"],
    feature_idx
)

#ex_result = build_cin(embeddings)
fm_result = build_fm(embeddings,feature_idx)
deep_result = build_deep(embeddings)
predict_weight = tf.get_variable(
        name='pre_weight',
        dtype=tf.float32,
        initializer=tf.glorot_normal_initializer(),
        shape=[2, 1])

out = tf.concat([fm_result,deep_result],axis=1)
out = tf.matmul(out,predict_weight)
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=labels)
loss = tf.reduce_mean(loss)
l2_reg = 0.01

loss = loss + tf.contrib.layers.l2_regularizer(l2_reg)(weights['concat_projection'])
for i in range(3):
    loss = loss + tf.contrib.layers.l2_regularizer(l2_reg)(weights["layer_%d" % i])

pre_result = tf.sigmoid(out)
optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
train_op = optimizer.minimize(loss)


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
            print t1
            #a = input()
        except tf.errors.OutOfRangeError:
            break

        if(cnt%10 == 0):
            val = tf.data.TextLineDataset("../../data/test_data.txt")
            val_dataset = val.map(parse_finish_line)
            val_dataset = val_dataset.batch(batch_size)
            val_dataset_iterator = val_dataset.make_one_shot_iterator()
            result = []
            true_lable = []
            while(True):
                try:
                    feat_index, feat_val, lable = sess.run(val_dataset_iterator.get_next())
                    feat_val = np.reshape(feat_val, newshape=[-1, 5, 1])
                    true_lable = true_lable + list(lable)
                    lable = np.reshape(lable, newshape=[-1, 1])
                    # print feat_index,feat_val,lable
                    pre = sess.run(pre_result,
                                     feed_dict={feature_idx: feat_index, feature_val: feat_val, labels: lable})
                    result = result + list(np.reshape(pre,newshape=[-1]))
                except tf.errors.OutOfRangeError:
                    break
            print "cnr:%d test_auc:%lf:",(cnt,roc_auc_score(true_lable,result))
        if(cnt == 2000):
            break

