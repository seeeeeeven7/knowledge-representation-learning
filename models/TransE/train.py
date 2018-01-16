import numpy
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
D = 300
BATCH_SIZE = 100
MARGIN = 1.0

# Load data

with open('../../data/WN18/wordnet-mlj12-train.txt', 'r') as trainfile:
	triples = trainfile.read().splitlines()

triples = list(map(lambda x : x.split('\t'), triples))

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

# H1R = tf.nn.embedding_lookup(ERs, X[:,0])
# R1R = tf.nn.embedding_lookup(RRs, X[:,1])
# T1R = tf.nn.embedding_lookup(ERs, X[:,2])
# H2R = tf.nn.embedding_lookup(ERs, Y[:,0])
# R2R = tf.nn.embedding_lookup(RRs, Y[:,1])
# T2R = tf.nn.embedding_lookup(ERs, Y[:,2])
loss = tf.reduce_mean(tf.nn.relu(tf.norm(H1R + R1R - T1R, axis = 1) + MARGIN - tf.norm(H2R + R2R - T2R, axis = 1)))

# Initialize
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Evaluation Model
Z = tf.placeholder(tf.int64, [None, ])
tf.concat(tf.expand_dims(X[:,1:], 0), tf.reshape()

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
		print(ERs[0].eval()   )

	if BATCH_INDEX % 1000 == 0:
		for triple in triples:
			# Predict head
			for h, hv in entity_index.items():
				pass

	BATCH_INDEX = BATCH_INDEX + 1