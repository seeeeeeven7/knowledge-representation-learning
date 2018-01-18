import matplotlib.pyplot as plt

losses_batch = []
losses_loss = []
evaluation_index = []
evaluation_meanrank = []
evaluation_hit10 = []
evaluation_hit5 = []
evaluation_hit2 = []
evaluation_hit1 = []

with open('log.txt', 'r') as infile:
	lines = infile.read().splitlines()
	index = 0
	while index < len(lines):
		print(lines[index].split(' '))
		[batch, loss] = lines[index].split(' ')
		if int(batch) <= 100000:
			losses_batch.append(int(batch))
			losses_loss.append(float(loss))
		else:
			break
		index = index + 1
		if len(losses_batch) % 10 == 1:
			[tmp, tmp, MeanRank, tmp, tmp, Hit10, tmp, tmp, Hit5, tmp, tmp, Hit2, tmp, tmp, Hit1] = lines[index].split(' ')
			evaluation_index.append(losses_batch[-1])
			evaluation_meanrank.append(float(MeanRank))
			evaluation_hit10.append(float(Hit10))
			evaluation_hit5.append(float(Hit5))
			evaluation_hit2.append(float(Hit2))
			evaluation_hit1.append(float(Hit1))
			index = index + 1

plt.figure()  
plt.plot(losses_batch, losses_loss)  
plt.savefig("loss.png") 

plt.figure()  
plt.plot(evaluation_index, evaluation_meanrank)  
plt.savefig("meanrank.png") 

plt.figure()  
plt.plot(evaluation_index[15:], evaluation_meanrank[15:])  
plt.savefig("meanrank_after150000.png") 

plt.figure()  
plt.plot(evaluation_index, evaluation_meanrank)  
plt.savefig("meanrank.png") 

plt.figure()  
plt.plot(evaluation_index, evaluation_hit10)  
plt.savefig("hit@10.png") 

plt.figure()  
plt.plot(evaluation_index, evaluation_hit5)  
plt.savefig("hit@5.png") 

plt.figure()  
plt.plot(evaluation_index, evaluation_hit2)  
plt.savefig("hit@2.png") 

plt.figure()  
plt.plot(evaluation_index, evaluation_hit1)  
plt.savefig("hit@1.png") 