import numpy
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
D = 300
BATCH_SIZE = 1000
MARGIN = 1.0
EPS = 1e-12

# Load data

with open('../../data/WN18/wordnet-mlj12-train.txt', 'r') as trainfile:
    triples = trainfile.read().splitlines()
with open('../../data/WN18/wordnet-mlj12-train.txt', 'r') as validfile:
    triples_valid = validfile.read().splitlines()

triples = list(map(lambda x : x.split('\t'), triples))
triples_valid = list(map(lambda x : x.split('\t'), triples_valid))

# prework

entity_index = {}
relation_index = {}

for triple in triples:
    if triple[0] not in entity_index:
        entity_index[triple[0]] = len(entity_index)
    if triple[2] not in entity_index:
        entity_index[triple[2]] = len(entity_index)
    if triple[1] not in relation_index:
        relation_index[triple[1]] = len(relation_index)

print(len(triples))
print(len(entity_index))

# Variable
ERs = tf.Variable(tf.random_uniform([len(entity_index), D], -6 / D, 6 / D, dtype=tf.float32))
RRs = tf.Variable(tf.random_uniform([len(relation_index), D], -6 / D, 6 / D, dtype=tf.float32))

# Input
X = tf.placeholder(tf.int64, [None, 3])
Y = tf.placeholder(tf.int64, [None, 3])

# Network
H1R = tf.nn.l2_normalize(tf.nn.embedding_lookup(ERs, X[:,0]), 1)
R1R = tf.nn.embedding_lookup(RRs, X[:,1])
T1R = tf.nn.l2_normalize(tf.nn.embedding_lookup(ERs, X[:,2]), 1)
H2R = tf.nn.l2_normalize(tf.nn.embedding_lookup(ERs, Y[:,0]), 1)
R2R = tf.nn.embedding_lookup(RRs, Y[:,1])
T2R = tf.nn.l2_normalize(tf.nn.embedding_lookup(ERs, Y[:,2]), 1)

loss = tf.reduce_mean(tf.nn.relu(tf.norm(H1R + R1R - T1R, axis = 1) + MARGIN - tf.norm(H2R + R2R - T2R, axis = 1)))

# Initialize
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Evaluation Model
x = tf.placeholder(tf.int64, [3])
HR = tf.nn.l2_normalize(tf.nn.embedding_lookup(ERs, x[0]), 0) # [300]
RR = tf.nn.embedding_lookup(RRs, x[1]) # [300]
TR = tf.nn.l2_normalize(tf.nn.embedding_lookup(ERs, x[2]), 0) # [300]
rankH = tf.count_nonzero(tf.nn.relu(EPS - tf.norm(tf.nn.l2_normalize(ERs, 1) - (TR - RR), axis = 1) + tf.norm(TR - RR - HR)))
rankT = tf.count_nonzero(tf.nn.relu(EPS - tf.norm(tf.nn.l2_normalize(ERs, 1) - (HR + RR), axis = 1) + tf.norm(TR - RR - HR)))

# Train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
BATCH_INDEX = 0
while True:
    batch_xs = []
    batch_ys = []
    for __ in range(BATCH_SIZE):
        index = numpy.random.randint(len(triples))
        batch_xs.append([entity_index[triples[index][0]], relation_index[triples[index][1]], entity_index[triples[index][2]]])
        if numpy.random.randint(2) == 0:
            batch_ys.append([entity_index[triples[index][0]], relation_index[triples[index][1]], numpy.random.randint(len(entity_index))])
        else:
            batch_ys.append([numpy.random.randint(len(entity_index)), relation_index[triples[index][1]], entity_index[triples[index][2]]])
    sess.run(train_step, feed_dict={X: batch_xs, Y:batch_ys})

    if BATCH_INDEX % 100 == 0:
        print(BATCH_INDEX, sess.run(loss, feed_dict={X: batch_xs, Y:batch_ys}))
        with open("loss.log", "a") as logfile:
            print(BATCH_INDEX, sess.run(loss, feed_dict={X: batch_xs, Y:batch_ys}), file = logfile)

    if BATCH_INDEX % 1000 == 0:
        rankSum = 0
        hit10 = 0
        hit5 = 0
        hit2 = 0
        hit1 = 0
        count = 0
        for triple in triples_valid:
            #if count == 1000:
            #    break
            if triple[0] not in entity_index:
                continue
            if triple[1] not in relation_index:
                continue
            if triple[2] not in entity_index:
                continue
            count = count + 1
            rankh = sess.run(rankH, feed_dict = {x: [entity_index[triple[0]], relation_index[triple[1]], entity_index[triple[2]]]})
            rankt = sess.run(rankT, feed_dict = {x: [entity_index[triple[0]], relation_index[triple[1]], entity_index[triple[2]]]})
            rankSum = rankSum + rankh + rankt
            if rankh <= 10:
                hit10 = hit10 + 1
            if rankh <= 5:
                hit5 = hit5 + 1
            if rankh <= 2:
                hit2 = hit2 + 1
            if rankh <= 1:
                hit1 = hit1 + 1
            if rankt <= 10:
                hit10 = hit10 + 1
            if rankt <= 5:
                hit5 = hit5 + 1
            if rankt <= 2:
                hit2 = hit2 + 1
            if rankt <= 1:
                hit1 = hit1 + 1

        print('MeanRank =', rankSum / (count * 2), 'Hit@10 =', hit10 / (count * 2), 'Hit@5 =', hit5 / (count * 2), 'Hit@2 =', hit2 / (count * 2), 'Hit@1 =', hit1 / (count * 2))
        with open("validation.log", "a") as logfile:
            print(BATCH_INDEX, rankSum / (count * 2), hit10 / (count * 2), hit5 / (count * 2), hit2 / (count * 2), hit1 / (count * 2), file = logfile)

    BATCH_INDEX = BATCH_INDEX + 1