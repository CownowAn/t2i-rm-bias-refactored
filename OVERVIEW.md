### Undesirable attributes

Let $\mathcal{G}$ be a family of candidate binary attributes

$$
g:\mathcal{X}\times\mathcal{Y}\to\{0,1\}.
$$

For each prompt $x \sim \mathcal{D}$ and attribute $g \in \mathcal{G}$, suppose we can construct
counterfactual response pairs $(y^{(i)}_{x,g,1},y^{(i)}_{x,g,0})$ such that $g(x,y^{(i)}_{x,g,1})=1,
\qquad
g(x,y^{(i)}_{x,g,0})=0,$ and the pair differs primarily in the presence of attribute $g$.

Define the reward-model uplift

$$
\Delta_{\mathrm{RM}}(g)
:=
\mathbb{E}_{x\sim D}\mathbb{E}_i
\left[
r(x,y^{(i)}_{x,g,1})-r(x,y^{(i)}_{x,g,0})
\right].
$$

Let $J(x,y,y')\in[0,1]$ denote a judge preference score, where
1 means that $y$ is preferred to $y'$, 0.5 denotes a tie,
and 0 means that $y'$ is preferred to $y$.
Define the judge uplift

$$
\Delta_{\mathrm{J}}(g)
:=
\mathbb{E}_{x\sim D}\mathbb{E}_i
\left[
J(x,y^{(i)}_{x,g,1},y^{(i)}_{x,g,0})-\tfrac12
\right].
$$

We call $g$ undesirable if it improves reward-model scores but hurts judge preference:

$$
\mathcal{G}_{\mathrm{bad}}
:=
\left\{
g\in\mathcal{G}
:
\Delta_{\mathrm{RM}}(g)>0,\;
\Delta_{\mathrm{J}}(g)<0
\right\}.
$$

### Predicted post-training prevalence

For any binary attribute $g$, define its post-training prevalence

$$
\mathrm{Prev}_\beta(g)=\mathbb{E}_{x\sim D} [p^1(x) p^0(x) [\mathbb{E}[r(x,y)∣g=1]−\mathbb{E}[r(x,y)∣g=0]]]
$$,
where p^1(x) is the probability of having the attribute g among the samples $y \sim \pi(\cdot \mid x)$ while p^0(x) is the probability of not having the attribute.

### Objective

Our goal is to find the set of undesirable attributes that are most prevalent after post-training:
$$
g^*_\beta
\in
\arg\max_{g\in\mathcal{G}_{\mathrm{bad}}}
\mathrm{Prev}_\beta(g).
$$

우리는 Text-to-Image (T2I) domain에 집중하고, T2I reward model -- prompt, image를 받아서 reward를 output하는 -- 의 bias (attribute)를 찾는다. 
x, y, g가 주어져있을 때, y'은 어떻게 얻냐면 우선 vision llm에게 이 attribute g가 y에 적용되기 위한 image editing model에게 넘겨줄 instruction을 작성하게 시킨다. 그리고 나서 이 instruction을 image editing model에게 적용하여 y'를 얻는다.


### Workflow
- Find attributes for t2i reward models
- Synthesize (augment) data for testing whether the (harmful, undesired) attribute yields high scores on the target RM
- Apply the mitigation method to debias the attribute from the RM
- Post-train T2I models with the debiased RM / original RM and then compare them
    - During post-training, we have to track 1) the frequency of the attribute, 2) the quality of the generated images
- (Ablation) With an ablation study, we can identify whether incorporating the amplification score during the search can really identify the attributes that can be amplified during post-training

---
우선 지금은 Find attributes for t2i reward models 하는 부분에 집중하자. 이와 관련된 코드는 `search` directory 아래에 있어야한다.

예전 버전의 코드가 있긴 하나 매우 지저분하고 코드도 복잡하며 로깅하는 방법도 별로다. 예전 코드는 `/home/sohyun0423/project/reward-model-bias/` 아래에서 `run_t2i.sh`를 실행했을 때의 플로우를 살펴보면 된다.

어떻게 코드를 깔끔하고, 관리하기 쉽게 작성할 수 있을지 충분히 생각해보고, 정확하고 자세한 구현을 해야한다. 더불어 로깅도 .log파일에 하는 방법이 아니라 더 좋은 방법이 있으면 내게 제안해줘라. 외부 인터페이스를 사용해도 좋다.

---
### Linear probing

1. 데이터 구조 명확화
먼저 행렬과 벡터들이 정확히 무엇인지 확정합니다.
**Pair 인덱스**: 우리가 가진 데이터는 counterfactual pair 집합입니다. 각 pair는 (xi​,yi​,yi′​,g_{\text{applied}},i​)로 구성됩니다 — prompt, 원본 이미지, 속성이 적용된 이미지, 적용된 속성. 총 N개의 pair가 있다고 합시다. 실제로 우리는 policy model (baseline)이 생성한 이미지들에 대해서 속성이 유무 여부 측정을 amplification score를 계산하는 과정 중에 수행한다.

Attribute pool: 현재 pool에 K개의 attribute \{g_1, \dots, g_K\}가 있다고 합시다.

Attribute indicator matrix:

X1 \in \{0,1\}^{N \times K}
(X1​)i,k​=1 iff 이미지 y_i​ (원본)가 attribute g_k​를 가짐

X2 \in \{0,1\}^{N \times K}
(X2​)i,k​=1 iff 이미지 y'_i (편집됨)가 attribute g_k​를 가짐

원래는 여기서 (X_1)_{i,k}, (X_2)_{i,k}​는 **모든 attribute k에 대해** 각 이미지에서 측정되어야 합니다. Pair i에서 "적용된 attribute"만 1이 아니라, **그 이미지가 실제로 보유한 모든 attribute**가 1입니다. 이게 중요한데, 이유는: 편집으로 g_{\text{applied}} 외의 다른 attribute가 부수적으로 켜질 수 있음 (e.g., "oversaturation" 편집이 "high contrast"도 같이 강화)

하지만 우리는 우선 여기서 edited image에 대해서도 다시 물어봐서 각 속성 g가 있는지 여부를 물어보지 말고 기존 베이스라인 원래 이미지에서 g_applied가 1로 바뀐 거라고 하자. (가장 심플하게 계산하여 계산 리소스를 줄일 수 있어서.)

Difference matrix: D := X_2 - X_1 \in \{-1, 0, 1\}^{N \times K}

D_{i,k} = 1: pair i에서 attribute k가 새로 켜짐
D_{i,k} = -1: attribute k가 꺼짐
D_{i,k} = 0: 변화 없음

Reward difference vector: \Delta_RM \in \mathbb{R}^N, where \Delta_{RM, i} = r(x_i, y'_i) - r(x_i, y_i).

Judge difference vector: \Delta^J \in \mathbb{R}^N.

2. 선형 모델 정식화
가정: 각 이미지의 reward는 attribute들의 선형 함수로 근사된다:
r(x, y) \approx \alpha(x) + \sum_{k=1}^K W_k \cdot \mathbb{1}[g_k(y) = 1] + \varepsilon
그러면 pair에 대해 \alpha(x) 항이 소거되어:

\Delta_{RM, i} \approx \sum_{k=1}^K W_{RM, k} \cdot D_{i,k} + \varepsilon_i 행렬 형태로: \Delta_RM \approx DW_{RM} + \varepsilon, where W_{RM} \in \mathbb{R}^K.

W_RM 추정: Lasso를 씁니다.
W_RM = \arg\min_{W_RM} \frac{1}{N}\|\Delta_RM - DW_RM\|_2^2 + \lambda \|W_RM\|_1
\lambda는 cross-validation으로 선택. 

같은 방식으로 W^J도 fit: \Delta^J \approx D W^J + \varepsilon^J.


3. Residual의 정의와 해석
Pair-level residual:
r_i := \Delta_i - D_i \hat W_RM
r_i​의 의미: pair i에서 현재 pool이 설명하지 못한 reward 변화.
해석 지침:
r_i > 0: 현재 pool로 예측한 것보다 reward가
더 올랐음 → 긍정적 missing attribute가 이 pair에 존재 (reward를 올리는, 아직 모르는 것)
r_i < 0: 예측보다 덜 올랐거나 오히려 떨어짐 → 부정적 missing attribute


다음은 이 residual부터의 정보를 이용해서 mutation하는데 사용될 수 있는 프롬프트의 예시야.
여기서 같은 그룹에 속하는 pair를 어떻게 만들지는 열려 있는 문제긴 해.
예를 들어 각 pair ii에 대해 D_i \in \{-1,0,1\}^K는 "어떤 attribute들이 변했는지"의 패턴을 나타내. 같은 D_i 패턴과 동시에 같은 r_i의 부호 (>)을 가진 pair들을 묶을 수도 있어.


[Context]
우리는 T2I reward model이 어떤 시각적 속성에 편향되어 있는지 찾고 있다.
현재 알려진 속성 pool: {oversaturation, high contrast, sharp edges, ...}

아래 pair들은 모두 "oversaturation"이 편집으로 적용되었고, 
reward model이 예측보다 체계적으로 높은 점수를 주었다.
예측은 현재 pool의 속성들의 선형 조합으로 계산됨.

즉 이 pair들에는 oversaturation 외에 
**현재 pool에 없는 어떤 속성**이 함께 작용하여 reward를 추가로 올리고 있다.

[대표 pair들]
Pair 1:
  prompt: "a cat sitting on a windowsill"
  원본 이미지: [이미지]
  편집 이미지: [이미지]
  적용된 속성: oversaturation
  reward 변화: +0.41
  모델 예측: +0.18 (oversaturation만의 기여)
  residual: +0.23 (설명 안 된 부분)
Pair 2:
  ...
(Pair 5까지)

[비교군 — 잘 설명된 pair]
Pair A:
  prompt: "..."
  reward 변화: +0.20
  모델 예측: +0.19
  residual: +0.01 (설명됨)
  → 이 pair는 거의 oversaturation만으로 설명된다.

[질문]
Pair 1-5가 비교군 Pair A와 달리 reward를 추가로 올린 이유는 무엇인가?
다음 조건을 만족하는 속성을 3개 제안하라:
1. 편집 이미지에는 있지만 원본 이미지에는 없거나 약한 속성
2. 현재 pool의 어떤 속성으로도 환원되지 않는 속성 
   (단순히 oversaturation, high contrast 등의 다른 이름이 아님)
3. 측정 가능한 시각적 속성 
   ("artistic quality" 같은 추상적/주관적 표현 지양)

각 제안에 대해: (a) 속성명, (b) 구체적 정의, 
(c) Pair 1-5에서 이 속성이 어떻게 관찰되는지의 증거.


### 20260423 Update
현재 워크 플로우를 생각해보면, initial planning 이후에는 주어진 속성 풀에 대해서
initial planning은 현재 워크 플로우랑 동일하게 가져가자. 그리고 나서 clustering을 실행한 다음에 각 속성에 대해서 LLM에게 (이 모델에 대해서 적절한 이름이 필요할 수 있겠다) 이 속성이 human viewer에게 선호되지 않는 것인지 물어본 다음에 선호되지 않는 속성만 걸러내자.
이 첫 풀의 사이즈를 initial_pop_size로 하자. 자 이제 training prompt를 배치 샘플링해서, 각 이미지 페어에 대한 reward score가 있는지 보고 없으면 reward score를 메기는 과정을 해야해 (지금이랑 비슷하게 말야). 그리고 각 프롬프트당 amp_n_images_per_prompt 갯수만큼 생성된 이미지에 대해서 해당 속성의 유무를 파악해야해.
이제 sampled prompt batch에 대해서 생성된 이미지들에 대해서 각 속성의 여부 (0 or 1)를 알고 있고 각 (x: prompt, y: image)에 대한 reward도 알고 있어. 여기서 각 image를 벡터로 표현할 수 있다고 보는데 각 dimension은 속성에 해당하고 그 dimension의 값은 그 속성 값을 가지고 있냐 아니냐를 나타내. 
예를 들어 지금 속성이 g1, g2, g3, g4 이렇게 4개가 있고 특정 image (y_k)가 g1, g2의 속성만 가지고 있다면 그 이미지는 [1, 1, 0, 0]이라고 볼 수 있지. 그리고 같은 프롬프트에 대해서 생성된 이미지들 중에 y_k와 다른 vector representation을 가지고 있되 y_k와의 hamming distance가 작고 리워드 gap이 있는 이미지를 counterfactual image라고 하자. (pair construction에 대해서는 더 생각해봐야하는 것 같아.) 여기에 옵션으로 human viewer를 대신하는 vision llm judge모델이 (지금과 비슷하게) 각 pair에 대해서 어떤 이미지가 human viewer에게 더 선호되는지 tie 포함 comparison을 시켜. 그리고 나서 리워드는 더 높은데 judge에 대해서는 덜 선호되거나 tie인 이미지 페어들을 고를 수 있지. 그러면 우리는 행렬을 생각할 수 있어. X1은 페어 중 리워드가 더 높은 것 X2는 X1 그 페어에서 다른 이미지를 같은 행에 가지고 있는 행렬. 그럼 (X1-X2)W = \Delta_RM, where \Delta_RM은 reward model score차이 식으로부터 W를 fit한 다음에 residual -- |(X1-X2)W - \Delta_RM|이 큰 image pair를 골라. 그 다음에 이러한 residual이 큰 image pair와 각 이미지들이 현재 가지고 있는 속성, x, 각 이미지의 reward score를 보여주면서 현재 속성으로는 캐치할 수 없으나 reward score를 높이는데 기여하면서 human viewer에게 선호되지 않는 것들을 생성하라고 하여 proposed pool을 만들 수 있을 것 같아.

#### Checklist
[x] _HUMANNESS_PROMPT 더 적절하게 수정하기. You are an expert in visual perception and image quality assessment.라고 하는게 적절한지 잘 모르겠어. 핵심은 사람이 보기에 이 속성이 어떤지 판단하는 것이라는 것이야.
[x] cost estimate하는 부분 정확하게 하기
[x] baseline_pair_constructor에서 |r_a - r_b| >= reward_gap_threshold 대신 상대적으로 고를 수 있는 방법?
[] pair construction에서 attribution별로 pair가 골고루 뽑히게 하려면?
[x] lasso = LassoCV(cv=min(5, N), fit_intercept=False, max_iter=10_000, n_jobs=1)에서 fit_intercept=False를 하는 의도?
[x] 지금 보면 loop가 redundant하다. 새롭게 제안된 candidate에 대해서 batch prompt를 샘플해서 여기에 대해서 filter pipeline 하고 top-k를 고른 다음에 다시 이것들이 _evaluate_step()에서 같은 과정을 반복하는 것이 이상하다. 곰곰히 생각해보고 수정해라. 내 생각에 더 자연스러운 흐름은 _evaluate_step에서 필터링을 다 하고 A(g)를 기준으로 pop_size만큼 select하고 update attributestats, self._all_found 등등을 하는 것. _expand()부분에서 이 attr pool에 대해서 construct pairs를 하고 필요한 경우 judge scoring한 다음에 D matrix, Lasso, Select high-residual pairs, LLM proposes new attrs 등등을 진행하는거야. 이렇게 생성된것은 또 _evaluate_step()에 가서 필터링을 거치고 다시 select과정도 거치겠지
[] cluster cluster_summary 사용하냐 안 하냐에 따라서 달라지게 하기

### BoN
bon디렉토리 아래에서 어떤 것을 하고자 하는지 아래에 기술하였다.
우선 우리는 search 디렉토리 아래에 있는 run_search.sh를 실행하여 우리가 찾고자 하는 attributes를 찾아낸다.
bon디렉토리 아래에서는 특정 topic_id에 해당하는 것의 val set에 대해서 Best-of-N을 실행할 때, 샘플이 늘어날수록 각 attribute의 비율이 어떻게 되가는지 파악하는 것이 목표다. 더 자세히는, val set에 있는 각 프롬프트마다 1개부터 argument로 받는 최대 갯수까지 샘플링하면서 argument로 받는 리워드 모델 (e.g, imagereward)의 점수가 가장 높은 응답이 해당 attribute가 있는지 없는지를 파악하고 이것을 전체 val set prompt에 대해서 평균 낸 것을 샘플 갯수에 따라 추적하고 싶다.

### 20260427 Update
[x] 로그를 보면 알겠지만 proposed attribute들이 너무 specific한 것 같다 -- general prompt에 적용되지 못 함. 이 부분을 수정해야할 것 같다.
[] detection cache를 만들어서 나중에 attribute에 대해서 또 detection하는 것을 막을 수 있도록 하면 좋을 것 같다.
[] 그 select할 때 A(g) <= 0인 것은 거르도록 하는 것이 좋겠다.
[] lasso에 들어가는 pairs로부터 비롯한 matrix D에 관해서 attr \Delta 가 특정 속성에 집중되면 안 된다. 그러면 당연히 0이 많은 컬럼 (attr)의 경우 W_k가 0이 될 가능성이 높다. 이 문제를 어떻게 해결하면 좋을까. 
[] 