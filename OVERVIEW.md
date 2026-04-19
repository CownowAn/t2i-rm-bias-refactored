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