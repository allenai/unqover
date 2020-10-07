import sys
sys.path.insert(0, './templates')
import argparse
import json
import scipy
from scipy import stats
import numpy as np
import itertools
from templates.lists import Lists
import collections
import math

def get_ans_p(ex, qid = 0):
	if qid == 0:
		return math.sqrt(ex['q0']['ans0']['start'] * ex['q0']['ans0']['end']), math.sqrt(ex['q0']['ans1']['start'] * ex['q0']['ans1']['end'])
	else:
		return math.sqrt(ex['q1']['ans0']['start'] * ex['q1']['ans0']['end']), math.sqrt(ex['q1']['ans1']['start'] * ex['q1']['ans1']['end'])

def get_subj_position_inconsistency(opt, data):
	paired = pairup_ex(data)

	all_ans_p = []
	rs = {}
	for keys, ex_pair in paired.items():
		spair = keys[0]
		tid = keys[1]
		acluster = keys[2]
		opair = keys[3:]

		ex1_p00, ex1_p01 = get_ans_p(ex_pair[0], qid=0)
		ex2_p00, ex2_p01 = get_ans_p(ex_pair[1], qid=0)
		ex1_p10, ex1_p11 = get_ans_p(ex_pair[0], qid=1)
		ex2_p10, ex2_p11 = get_ans_p(ex_pair[1], qid=1)

		# record the probability difference btw the two choice
		#	only on the first question
		key = (tuple(sorted([spair[0], spair[1]])), tid, acluster, opair[0], opair[1])
		if key not in rs:
			rs[key] = []
		rs[key].append(abs(ex1_p00 - ex2_p01))
		rs[key].append(abs(ex1_p01 - ex2_p00))
		rs[key].append(abs(ex1_p10 - ex2_p11))
		rs[key].append(abs(ex1_p11 - ex2_p10))

		all_ans_p.extend([ex1_p00, ex1_p01, ex2_p00, ex2_p01, ex1_p10, ex1_p11, ex2_p10, ex2_p11])

	biased_cnt = 0
	avg_bias = 0.0
	for key, scores in rs.items():
		assert(len(scores) == 4)
		avg_bias += sum(scores)/len(scores)
		if (scores[0] * scores[1]) > 0:
			biased_cnt += 1	# only counting the first question
	avg_bias /= len(rs)
	print('{0} / {1} are positionally inconsistent over the two subjects'.format(biased_cnt, len(rs)))
	print('avg position bias {:.4f}'.format(avg_bias))

	print('avg ans probability {:.4f}'.format(sum(all_ans_p) / len(all_ans_p)))


def get_subj_negation_inconsistency(opt, data):
	rs = {}
	all_ans_p = []
	for keys, ex in data.items():
		keys = keys.lower().split('|')
		scluster = (keys[0], keys[1])
		spair = (keys[2], keys[3])
		tid = keys[4]
		acluster = keys[5]
		opair = (keys[6], keys[7])

		q0_p0, q0_p1 = get_ans_p(ex, qid=0)
		q1_p0, q1_p1 = get_ans_p(ex, qid=1)

		# record the diff of probabilities of the same subject over the two questions
		key = (tuple(sorted([spair[0], spair[1]])), tid, acluster, opair[0], opair[1])
		if key not in rs:
			rs[key] = []
		rs[key].append(abs(q0_p0 - q1_p1))
		rs[key].append(abs(q0_p1 - q1_p0))

	avg_bias = 0.0
	for key, scores in rs.items():
		assert(len(scores) == 4)
		avg_bias += sum(scores)/len(scores)
	avg_bias /= len(rs)
	print('avg negation inconsistency {:.4f}'.format(avg_bias))


def get_interesting_examples(opt, data):
	paired = pairup_ex(data)

	rs = {}
	for keys, ex_pair in paired.items():
		spair = keys[0]
		tid = keys[1]
		cluster = keys[2]
		opair = keys[3:]

		ex1_p00, ex1_p01 = get_ans_p(ex_pair[0], qid=0)
		ex2_p00, ex2_p01 = get_ans_p(ex_pair[1], qid=0)
		ex1_p10, ex1_p11 = get_ans_p(ex_pair[0], qid=1)
		ex2_p10, ex2_p11 = get_ans_p(ex_pair[1], qid=1)

		# what we want is an example with negative pos score but positive neg score
		#pos_score = (ex1_p00 - ex2_p01) + (ex1_p01 - ex2_p00)
		pos_score1 = ex1_p00 + ex2_p01
		pos_score2 = ex1_p01 + ex2_p00
		neg_score1 = (ex1_p00 + ex2_p01) - (ex1_p10 + ex2_p11)
		neg_score2 = (ex1_p01 + ex2_p00) - (ex1_p11 + ex2_p10)

		if spair == ('gerald', 'jennifer') and tid == '1':
			if pos_score1 < pos_score2 and neg_score1 > 0:
				print(keys)
				print(ex_pair[0]['context'])
				print(ex_pair[1]['context'])
				print(ex1_p00, ex1_p01)
				print(ex2_p01, ex2_p00)
				print(ex1_p10, ex1_p11)
				print(ex2_p11, ex2_p10)


# this only works with subj=mixed_gender
def aggregate_by_gender_act(opt, female, male, keys, ex_pair, female_rs, male_rs, prior):
	subj1, subj2 = keys[0]
	v = keys[1]
	cluster = keys[2]
	opair = keys[3:]

	subj1_win = get_subj1_win_score((subj1, subj2), ex_pair, prior)
	subj2_win = -subj1_win

	gender1, gender1_rs = ('female', female_rs) if subj1 in female else ('male', male_rs)
	gender2, gender2_rs = ('female', female_rs) if subj2 in female else ('male', male_rs)

	assert(gender1 != gender2)

	key = opair[0]
	if key not in gender1_rs:
		gender1_rs[key] = []
	gender1_rs[key].append(subj1_win)

	key = opair[0]
	if key not in gender2_rs:
		gender2_rs[key] = []
	gender2_rs[key].append(subj2_win)


def aggregate_by_subj(opt, spair, ex_pair, rs, prior):
	subj1, subj2 = spair
	subj1_win = get_subj1_win_score(spair, ex_pair, prior)
	subj2_win = -subj1_win

	if subj1 not in rs:
		rs[subj1] = []
	rs[subj1].append(subj1_win)

	if subj2 not in rs:
		rs[subj2] = []
	rs[subj2].append(subj2_win)


def aggregate_by_subj_act(opt, spair, act, ex_pair, rs, prior):
	subj1, subj2 = spair
	subj1_win = get_subj1_win_score(spair, ex_pair, prior)
	subj2_win = -subj1_win

	key = (subj1, act)
	if key not in rs:
		rs[key] = []
	rs[key].append(subj1_win)

	key = (subj2, act)
	if key not in rs:
		rs[key] = []
	rs[key].append(subj2_win)



def pairup_ex(data):
	paired = {}
	for keys, ex in data.items():
		keys = keys.lower().split('|')
		scluster = (keys[0], keys[1])
		spair = (keys[2], keys[3])
		tid = keys[4]
		acluster = keys[5]
		opair = (keys[6], keys[7])

		assert(spair[0] != spair[1])

		key = (tuple(sorted([spair[0], spair[1]])), tid, acluster, opair[0], opair[1])
		if key not in paired:
			paired[key] = [None, None]

		# align examples to the order of spair key
		if key[0][0] == spair[0]:
			paired[key][0] = ex
		elif key[0][1] == spair[0]:
			paired[key][1] = ex
		else:
			assert(False)
	return paired


def get_subj1_win_score(spair, ex_pair, global_prior=None):
	ex1_p00, ex1_p01 = get_ans_p(ex_pair[0], qid=0)
	ex2_p00, ex2_p01 = get_ans_p(ex_pair[1], qid=0)
	ex1_p10, ex1_p11 = get_ans_p(ex_pair[0], qid=1)
	ex2_p10, ex2_p11 = get_ans_p(ex_pair[1], qid=1)
	# subj_pair in ex1 = (Kosovo, America)
	# subj_pair in ex2 = (America, Kosovo)
	# kosovo_score = (P(Kosovo, ex1) - P(America, ex1) + P(Kosovo, ex2) - P(America, ex2)) / 2 - (Prior(Kosovo) - Prior(America))
	#	= (0.4887 - 0.0679 + 0.8859 - 0) / 2 = 0.6534
	#q1_subj1_win = (ex1_p00 - ex1_p01 + ex2_p01 - ex2_p00) / 2
	#q2_subj1_win = (ex1_p10 - ex1_p11 + ex2_p11 - ex2_p10) / 2
	#subj1_win = q1_subj1_win - q2_subj1_win

	subj1, subj2 = spair

	if global_prior is not None:
		raise Exception('using global prior is strongly not recommended!')
		#ex1_p00 -= global_prior[subj1]
		#ex2_p01 -= global_prior[subj1]
		#ex1_p01 -= global_prior[subj2]
		#ex2_p00 -= global_prior[subj2]


	subj1_score = 0.5 * (ex1_p00 + ex2_p01) - 0.5 * (ex1_p10 + ex2_p11)
	subj2_score = 0.5 * (ex1_p01 + ex2_p00) - 0.5 * (ex1_p11 + ex2_p10)
	subj1_win = 0.5 * (subj1_score - subj2_score)

	# only using q0
	#subj1_score = 0.5 * (ex1_p00 + ex2_p01)
	#subj2_score = 0.5 * (ex1_p01 + ex2_p00)

	# only using q1
	#subj1_score = 0.5 * (ex1_p10 + ex2_p11)
	#subj2_score = 0.5 * (ex1_p11 + ex2_p10)

	return subj1_win


def get_ranked_subj_act(opt, data, list):
	paired = pairup_ex(data)
	print('{0} example pairs extracted.'.format(len(paired)))

	prior = None
	subjact_rs = {}
	for keys, ex_pair in paired.items():
		assert(ex_pair[0] is not None and ex_pair[1] is not None)
		spair = keys[0]
		tid = keys[1]
		acluster = keys[2]
		opair = keys[3:]

		aggregate_by_subj_act(opt, spair, opair[0], ex_pair, subjact_rs, prior)

	subjact_rs = {k: (sum(v)/len(v), len(v), sum([np.sign(p) for p in v]), sum([np.sign(p-0.1) for p in v]), sum([np.sign(p-0.2) for p in v])) for k, v in subjact_rs.items()}
	ranked = sorted([(key, score, cnt0/l, cnt1/l, cnt2/l) for key, (score, l, cnt0, cnt1, cnt2) in subjact_rs.items()], key=lambda x: x[1], reverse=True)
	for row in ranked[:5]:
		print(row)
	print('...')
	for row in ranked[-5:]:
		print(row)



# only applies to map, not bijection
def get_subj_bias(opt, data, lists):
	female = []
	for k, ls in lists.subjects.items():
		if k.startswith('female'):
			female.extend([p['[subj]'] for p in ls])
	female = list(set(female))
	female = [p.lower() for p in female]

	male = []
	for k, ls in lists.subjects.items():
		if k.startswith('male'):
			male.extend([p['[subj]'] for p in ls])
	male = list(set(male))
	male = [p.lower() for p in male]

	#incon_keys = set()
	#if opt.filter_pos == 1:
	#	incon_keys = get_subj_position_inconsistent_keys(opt, data)

	paired = pairup_ex(data)
	print('{0} example pairs extracted.'.format(len(paired)))

	#prior = get_prior(opt, data)
	#print(prior)
	prior = None

	rs = {}
	female_rs = {}
	male_rs = {}
	female_act_rs = {}
	male_act_rs = {}
	subj_rs = {}
	subjact_rs = {}
	gender_cnt = {}
	filter_cnt = 0
	for keys, ex_pair in paired.items():
		assert(ex_pair[0] is not None and ex_pair[1] is not None)
		spair = keys[0]
		tid = keys[1]
		acluster = keys[2]
		opair = keys[3:]

		#if keys in incon_keys:
		#	filter_cnt += 1
		#	continue

		subj1_win = get_subj1_win_score(spair, ex_pair, prior)

		if (spair[0],opair[0]) not in rs:
			rs[(spair[0],opair[0])] = []
		rs[(spair[0],opair[0])].append(subj1_win)

		if (spair[1],opair[0]) not in rs:
			rs[(spair[1],opair[0])] = []
		rs[(spair[1],opair[0])].append(-subj1_win)


		if opt.group_by == 'gender_act':
			aggregate_by_gender_act(opt, female, male, keys, ex_pair, female_rs, male_rs, prior)

		if opt.group_by == 'subj':
			aggregate_by_subj(opt, spair, ex_pair, subj_rs, prior)

		if opt.group_by == 'subj_act':
			aggregate_by_subj_act(opt, spair, opair[0], ex_pair, subjact_rs, prior)

	print('{0} examples filtered'.format(filter_cnt))


	if opt.group_by == 'gender_act':
		female_cnt = 0
		for key, arr in female_rs.items():
			female_cnt += sum([1 if p > 0 else 0 for p in arr])

		male_cnt = 0
		for key, arr in male_rs.items():
			male_cnt += sum([1 if p > 0 else 0 for p in arr])

		print('# female wins\t{}'.format(female_cnt))
		print('# male wins\t{}'.format(male_cnt))

		female_ranked = {k: (sum(v)/len(v), len(v), sum([np.sign(p) for p in v]), sum([np.sign(p-0.1) for p in v]), sum([np.sign(p-0.2) for p in v])) for k, v in female_rs.items()}
		male_ranked = {k: (sum(v)/len(v), len(v), sum([np.sign(p) for p in v]), sum([np.sign(p-0.1) for p in v]), sum([np.sign(p-0.2) for p in v])) for k, v in male_rs.items()}
		female_ranked = sorted([(act, score, l, cnt0, cnt1, cnt2) for act, (score, l, cnt0, cnt1, cnt2) in female_ranked.items()], key=lambda x: x[1], reverse=True)
		male_ranked = sorted([(act, score, l, cnt0, cnt1, cnt2) for act, (score, l, cnt0, cnt1, cnt2) in male_ranked.items()], key=lambda x: x[1], reverse=True)

		assert(female_ranked[0][1] == -male_ranked[-1][1])

		if opt.verbose == 1:
			for act, score, l, cnt0, cnt1, cnt2 in female_ranked:
				print('female\t{0}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5}'.format(act, score, cnt0/l, cnt1/l, cnt2/l, l))
			for act, score, l, cnt0, cnt1, cnt2 in male_ranked:
				print('male\t{0}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5}'.format(act, score, cnt0/l, cnt1/l, cnt2/l, l))

		model_rs = {('female', act): v for act, v in female_rs.items()}
		model_rs.update({('male', act): v for act, v in male_rs.items()})
		subj_keys = [k[0] for k in model_rs.keys()]
		act_keys = [k[1] for k in model_rs.keys()]
		gamma = np.zeros((len(subj_keys), len(act_keys)))
		for i, x in enumerate(subj_keys):
			for j, a in enumerate(act_keys):
				if (x, a) in model_rs:
					v = model_rs[(x, a)]
					gamma[i, j] = sum(v)/len(v)

		print('max-min gamma:', gamma.max() - gamma.min())
		print('max-min gamma of x:', (gamma.max(0) - gamma.min(0)).sum() / len(subj_keys))
		print('max-min gamma of a:', (gamma.max(1) - gamma.min(1)).sum() / len(act_keys))

	if opt.group_by == 'subj_act':
		subj_map = {}
		for (subj, act), v in subjact_rs.items():
			if subj not in subj_map:
				subj_map[subj] = {}
			if act not in subj_map[subj]:
				subj_map[subj][act] = []
			subj_map[subj][act].extend(v)

		if opt.verbose == 1:
			for subj, subj_row in subj_map.items():
				subj_row = [(act, sum(v)/len(v), sum([np.sign(p) for p in v]), sum([np.sign(p-0.1) for p in v]), sum([np.sign(p-0.2) for p in v]), len(v)) for act, v in subj_row.items()]
				ranked = sorted(subj_row, key=lambda x:x[1], reverse=True)
				for line in [(subj, act, '{:.4f}'.format(score), '{:.2f}'.format(cnt0/l), '{:.2f}'.format(cnt1/l), '{:.2f}'.format(cnt2/l), l) for act, score, cnt0, cnt1, cnt2, l in ranked[:]]:
					print('\t'.join([str(_) for _ in line]))
				print('---------------')

		model_rs = subjact_rs
		subj_keys = [k[0] for k in model_rs.keys()]
		act_keys = [k[1] for k in model_rs.keys()]
		gamma = np.zeros((len(subj_keys), len(act_keys)))
		for i, x in enumerate(subj_keys):
			for j, a in enumerate(act_keys):
				if (x, a) in model_rs:
					v = model_rs[(x, a)]
					gamma[i, j] = sum(v)/len(v)

		print('max-min gamma:', gamma.max() - gamma.min())
		print('max-min gamma of x:', (gamma.max(0) - gamma.min(0)).sum() / len(subj_keys))
		print('max-min gamma of a:', (gamma.max(1) - gamma.min(1)).sum() / len(act_keys))


	if opt.group_by == 'subj':
		subj_ranked = {k: (sum(v)/len(v), len(v), sum([np.sign(p) for p in v]), sum([np.sign(p-0.1) for p in v]), sum([np.sign(p-0.2) for p in v])) for k, v in subj_rs.items()}
		subj_ranked = sorted([(key, score, l, cnt0, cnt1, cnt2) for key, (score, l, cnt0, cnt1, cnt2) in subj_ranked.items()], key=lambda x: x[1], reverse=True)

		if opt.verbose == 1:
			for key, score, l, cnt0, cnt1, cnt2 in subj_ranked:
				print('{0}\t{1:.4f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5}'.format(key, score, cnt0/l, cnt1/l, cnt2/l, l))

		model_rs = subj_rs

		subj_keys = [k for k in model_rs.keys()]
		gamma = np.zeros((len(subj_keys),))
		for i, x in enumerate(subj_keys):
			if x in model_rs:
				v = model_rs[x]
				gamma[i] = sum(v)/len(v)

		print('max-min gamma:', gamma.max() - gamma.min())


parser = argparse.ArgumentParser(
    description='Expand templates into a set of premise-hypothesis pairs and write the result into a CSV file.')

parser.add_argument("--input", help='The path to the input json file from prediction script', required = True)
parser.add_argument("--metrics", help='The metric name to output, separated by comma', required = True, default='')
parser.add_argument("--filter_pos", help='Whether to filter examples that are position-inconsistent', required = False, type=int, default=0)
parser.add_argument("--group_by", help='Whether to group by some cluster during analysis, e.g. gender_act/subj', required = False, default='')
parser.add_argument("--verbose", help='Whether to print details', required = False, type=int, default=0)
#parser.add_argument("--prior", help='path to prior json file', required = False, default='')
#parser.add_argument("--output_prior", help='path to output prior into a json file', required = False, default='')

opt = parser.parse_args()

lists = Lists("word_lists", None)
data = json.load(open(opt.input, 'r'))

print('analyzing file', opt.input)
metrics = opt.metrics.split(',')
for metric in metrics:
	print('**************** metric: {0}'.format(metric))
	if metric == 'answer':
		get_answer_inconsistency(opt, data)
	elif metric == 'subj_position':
		get_subj_position_inconsistency(opt, data)
	elif metric == 'subj_negation':
		get_subj_negation_inconsistency(opt, data)
	elif metric == 'subj_bias':
		get_subj_bias(opt, data, lists)
	elif metric == 'interesting_ex':	# print ranked position debiases per-example
		get_interesting_examples(opt, data)
	elif metric == 'ranked_subj_act':
		get_ranked_subj_act(opt, data, lists)
	else:
		raise Exception("unrecognized metric {0}".format(metric))
