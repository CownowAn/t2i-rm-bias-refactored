# acc_pool	(K,) object	attribute 이름 (D의 column label)
# D	(N, K) int8	attribute 차이 행렬
# delta_rm_vec	(N,) float32	pair별 RM gap
# W_rm	(K,) float32	regression 계수
# residuals	(N,) float32	pair별 잔차
# reg_intercept	scalar	bias term
# reg_alpha	scalar	CV로 선택된 정규화 강도
# reg_l1_ratio	scalar	L1/(L1+L2) 비율

path = "/home/sohyun0423/project/t2i-rm-bias/outputs/search/20260428-113420/bp_reg_step0_topic0.npz"
path = "/home/sohyun0423/project/t2i-rm-bias/outputs/search/20260428-113420/bp_reg_step1_topic0.npz"

import numpy as np
data = np.load(path, allow_pickle=True)

import pdb; pdb.set_trace()